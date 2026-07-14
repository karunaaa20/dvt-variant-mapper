import streamlit as st
import requests
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import re
from Bio.Align import substitution_matrices

st.set_page_config(page_title="Variant Interaction Risk Mapper", layout="wide")

BLOSUM62 = substitution_matrices.load("BLOSUM62")

AA_3TO1 = {
    'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E',
    'Gly':'G','His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F',
    'Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V'
}

OPEN_TARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"
QUICKGO_API = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GNOMAD_API = "https://gnomad.broadinstitute.org/api"
STRING_API = "https://string-db.org/api/json/network"

ROLE_SYNERGY = {
    ("promotes", "suppresses"): 1.5,
    ("suppresses", "promotes"): 1.5,
    ("promotes", "promotes"): 1.1,
    ("suppresses", "suppresses"): 1.1,
    ("promotes", "substrate"): 1.3,
    ("substrate", "promotes"): 1.3,
    ("suppresses", "substrate"): 1.2,
    ("substrate", "suppresses"): 1.2,
}
DEFAULT_SYNERGY = 1.0


# ---------- Core pipeline functions (same logic as your notebook) ----------

def search_disease_id(disease_name):
    query = """
    query search($q: String!) {
      search(queryString: $q, entityNames: ["disease"]) {
        hits { id name entity }
      }
    }
    """
    r = requests.post(OPEN_TARGETS_API, json={"query": query, "variables": {"q": disease_name}})
    r.raise_for_status()
    hits = r.json()["data"]["search"]["hits"]
    if not hits:
        raise ValueError(f"No disease match found for '{disease_name}'")
    return hits[0]["id"], hits[0]["name"]


def get_genes(disease_name, top_n=15, min_score=0.1):
    disease_id, matched_name = search_disease_id(disease_name)
    query = """
    query assoc($efoId: String!) {
      disease(efoId: $efoId) {
        associatedTargets(page: {index: 0, size: 100}) {
          rows { score target { approvedSymbol } }
        }
      }
    }
    """
    r = requests.post(OPEN_TARGETS_API, json={"query": query, "variables": {"efoId": disease_id}})
    r.raise_for_status()
    rows = r.json()["data"]["disease"]["associatedTargets"]["rows"]
    genes = [(row["target"]["approvedSymbol"], row["score"]) for row in rows if row["score"] >= min_score]
    genes = sorted(genes, key=lambda x: -x[1])[:top_n]
    return pd.DataFrame(genes, columns=["gene", "association_score"]), matched_name


def get_uniprot_id(gene_symbol, organism="9606"):
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": f"gene:{gene_symbol} AND organism_id:{organism} AND reviewed:true", "format": "json", "size": 1}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return None
    results = r.json().get("results", [])
    return results[0]["primaryAccession"] if results else None


def suggest_role(gene_symbol, process_keyword="coagulation"):
    try:
        uniprot_id = get_uniprot_id(gene_symbol)
        if not uniprot_id:
            return "unclear"
        r = requests.get(QUICKGO_API, params={"geneProductId": uniprot_id, "limit": 100})
        if r.status_code != 200:
            return "unclear"
        results = r.json().get("results", [])
        if not isinstance(results, list):
            return "unclear"
        pos, neg = 0, 0
        for res in results:
            if not isinstance(res, dict):
                continue
            name = str(res.get("goName", "")).lower()
            if process_keyword in name and "positive regulation" in name:
                pos += 1
            elif process_keyword in name and "negative regulation" in name:
                neg += 1
        if pos > neg:
            return "promotes"
        elif neg > pos:
            return "suppresses"
        return "unclear"
    except Exception:
        return "unclear"

def get_clinvar_vus(gene_symbol, retmax=30):
    term = f'{gene_symbol}[gene] AND single_gene[prop] AND ("uncertain significance"[Clinical_Significance])'
    r = requests.get(f"{NCBI_EUTILS}/esearch.fcgi", params={
        "db": "clinvar", "term": term, "retmax": retmax, "retmode": "json"
    })
    r.raise_for_status()
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return pd.DataFrame(columns=["gene", "variant_id", "hgvs_p", "protein_pos", "ref_aa", "alt_aa"])
    r2 = requests.get(f"{NCBI_EUTILS}/esummary.fcgi", params={
        "db": "clinvar", "id": ",".join(ids), "retmode": "json"
    })
    r2.raise_for_status()
    summaries = r2.json().get("result", {})
    rows = []
    for vid in ids:
        title = summaries.get(vid, {}).get("title", "")
        match = re.search(r'p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})', title)
        if not match:
            continue
        ref3, pos, alt3 = match.groups()
        ref1 = AA_3TO1.get(ref3.capitalize())
        alt1 = AA_3TO1.get(alt3.capitalize())
        if not ref1 or not alt1:
            continue
        rows.append({"gene": gene_symbol, "variant_id": vid, "hgvs_p": title,
                      "protein_pos": int(pos), "ref_aa": ref1, "alt_aa": alt1})
    return pd.DataFrame(rows)


def get_gnomad_af(gene_symbol, protein_pos, ref_aa, alt_aa):
    try:
        query = """
        query GeneVariants($geneSymbol: String!) {
          gene(gene_symbol: $geneSymbol, reference_genome: GRCh38) {
            variants(dataset: gnomad_r4) { pos hgvsp exome { af } genome { af } }
          }
        }
        """
        r = requests.post(GNOMAD_API, json={"query": query, "variables": {"geneSymbol": gene_symbol}})
        if r.status_code != 200:
            return None
        variants = r.json().get("data", {}).get("gene", {}).get("variants", [])
        for v in variants:
            hgvsp = v.get("hgvsp", "") or ""
            if str(protein_pos) in hgvsp and ref_aa in hgvsp:
                exome = v.get("exome") or {}
                genome = v.get("genome") or {}
                return exome.get("af") if exome.get("af") is not None else genome.get("af")
        return None
    except Exception:
        return None


def get_functional_sites(uniprot_id):
    if not uniprot_id:
        return []
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json")
    if r.status_code != 200:
        return []
    sites = []
    for feat in r.json().get("features", []):
        if feat.get("type") in ("Active site", "Binding site", "Site"):
            loc = feat.get("location", {}).get("start", {}).get("value")
            if loc:
                sites.append(loc)
    return sites


def blosum_severity(ref_aa, alt_aa):
    try:
        raw_score = BLOSUM62[ref_aa, alt_aa]
    except KeyError:
        return 0.5
    severity = 1 - ((raw_score + 4) / 15)
    return max(0, min(1, severity))


def proximity_weight(protein_pos, functional_sites):
    if not functional_sites:
        return 0.3
    min_dist = min(abs(protein_pos - site) for site in functional_sites)
    return 1 / (min_dist + 1)


def get_network_scores(gene_list, species=9606, confidence=0.7):
    params = {"identifiers": "%0d".join(gene_list), "species": species,
              "required_score": int(confidence * 1000)}
    r = requests.get(STRING_API, params=params)
    r.raise_for_status()
    edges = r.json()
    G = nx.Graph()
    G.add_nodes_from(gene_list)
    for e in edges:
        G.add_edge(e["preferredName_A"], e["preferredName_B"], weight=e["score"])

    def normalize(d):
        vals = list(d.values())
        if not vals or max(vals) == min(vals):
            return {k: 0.5 for k in d}
        return {k: (v - min(vals)) / (max(vals) - min(vals)) for k, v in d.items()}

    bet = normalize(nx.betweenness_centrality(G))
    deg = normalize(nx.degree_centrality(G))
    scores = {g: 0.5 * bet.get(g, 0) + 0.5 * deg.get(g, 0) for g in gene_list}
    return scores, G


def get_synergy(role_a, role_b):
    return ROLE_SYNERGY.get((role_a, role_b), DEFAULT_SYNERGY)


def pathway_distance_factor(gene_a, gene_b, G):
    try:
        dist = nx.shortest_path_length(G, gene_a, gene_b)
        return 1 / (dist + 1)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return 0.1


# ---------------------------- Streamlit UI ----------------------------

st.title("🧬 Variant Interaction Risk Mapper")
st.caption(
    "Type a disease name. The tool pulls associated genes, fetches candidate "
    "variants of uncertain significance, scores each one structurally and by "
    "network importance, and ranks which variant PAIRS are most likely to "
    "interact and compound risk — without needing any labeled training data."
)

with st.sidebar:
    st.header("Input")
    disease_name = st.text_input("Disease name", value="deep vein thrombosis")
    top_n_genes = st.slider("Number of genes to include", 5, 25, 15)
    af_threshold = st.number_input("Max allele frequency (rarity filter)", value=0.01, format="%.4f")
    top_fraction = st.slider("Fraction of top variants used for pairing", 0.1, 1.0, 0.5)
    run_button = st.button("Run pipeline", type="primary")

if run_button:
    progress = st.progress(0, text="Fetching gene panel...")

    # Step 1
    gene_panel_df, matched_name = get_genes(disease_name, top_n=top_n_genes)
    st.subheader(f"Gene panel for: {matched_name}")
    st.dataframe(gene_panel_df, use_container_width=True)
    progress.progress(15, text="Suggesting gene roles...")

    # Step 2
    suggested_roles = {}
    for gene in gene_panel_df["gene"]:
        suggested_roles[gene] = suggest_role(gene)
        time.sleep(0.2)
    role_display_df = pd.DataFrame(list(suggested_roles.items()), columns=["gene", "auto_suggested_role"])
    st.subheader("Auto-suggested gene roles (from GO annotations)")
    st.caption("These are a starting point, not ground truth — shown transparently rather than hidden.")
    st.dataframe(role_display_df, use_container_width=True)
    progress.progress(30, text="Fetching candidate variants (this takes a minute)...")

    # Step 3
    all_variants = []
    for i, gene in enumerate(gene_panel_df["gene"]):
        df = get_clinvar_vus(gene)
        for _, row in df.iterrows():
            af = get_gnomad_af(gene, row["protein_pos"], row["ref_aa"], row["alt_aa"])
            row_dict = row.to_dict()
            row_dict["gnomad_af"] = af
            if af is None or af < af_threshold:
                all_variants.append(row_dict)
        time.sleep(0.3)
        progress.progress(30 + int(30 * (i + 1) / len(gene_panel_df)), text=f"Fetched variants for {gene}...")
    variants_df = pd.DataFrame(all_variants)
    st.subheader(f"Candidate rare VUS variants ({len(variants_df)} found)")
    st.dataframe(variants_df, use_container_width=True)

    if variants_df.empty:
        st.warning("No variants found for this gene panel — try a different disease or loosen the AF threshold.")
        st.stop()

    progress.progress(65, text="Fetching structural functional sites...")

    # Step 4 + 5
    functional_sites_by_gene = {}
    for gene in gene_panel_df["gene"]:
        uid = get_uniprot_id(gene)
        functional_sites_by_gene[gene] = get_functional_sites(uid)
        time.sleep(0.2)

    variants_df["structural_score"] = variants_df.apply(
        lambda row: blosum_severity(row["ref_aa"], row["alt_aa"]) *
                    proximity_weight(row["protein_pos"], functional_sites_by_gene.get(row["gene"], [])),
        axis=1
    )
    progress.progress(75, text="Scoring network importance...")

    # Step 6
    network_scores, G = get_network_scores(list(gene_panel_df["gene"]))

    # Step 7
    variants_df["network_score"] = variants_df["gene"].map(network_scores).fillna(0.1)
    variants_df["ptrs_single"] = variants_df["structural_score"] * variants_df["network_score"]
    variants_df = variants_df.sort_values("ptrs_single", ascending=False)

    st.subheader("Ranked single-variant risk scores")
    st.dataframe(variants_df, use_container_width=True)
    progress.progress(85, text="Scoring variant pair interactions...")

    # Step 8
    n_keep = max(1, int(len(variants_df) * top_fraction))
    top_variants = variants_df.head(n_keep).reset_index(drop=True)
    pairs = []
    for i in range(len(top_variants)):
        for j in range(i + 1, len(top_variants)):
            v_a, v_b = top_variants.iloc[i], top_variants.iloc[j]
            if v_a["gene"] == v_b["gene"]:
                continue
            role_a = suggested_roles.get(v_a["gene"], "unclear")
            role_b = suggested_roles.get(v_b["gene"], "unclear")
            synergy = get_synergy(role_a, role_b)
            path_factor = pathway_distance_factor(v_a["gene"], v_b["gene"], G)
            combined = v_a["ptrs_single"] + v_b["ptrs_single"] + synergy * path_factor
            pairs.append({
                "gene_a": v_a["gene"], "variant_a": v_a["hgvs_p"],
                "gene_b": v_b["gene"], "variant_b": v_b["hgvs_p"],
                "role_a": role_a, "role_b": role_b,
                "synergy_multiplier": synergy, "combined_score": combined
            })
    pairs_df = pd.DataFrame(pairs).sort_values("combined_score", ascending=False)

    st.subheader("🔺 Top predicted high-risk variant combinations")
    st.dataframe(pairs_df.head(30), use_container_width=True)
    progress.progress(95, text="Building network visualization...")

    # Step 10 - visualization
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(G, seed=42)
    structural_by_gene = variants_df.groupby("gene")["structural_score"].max().to_dict()
    node_sizes = [3000 * network_scores.get(n, 0.1) + 200 for n in G.nodes()]
    node_colors = [structural_by_gene.get(n, 0.3) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                            cmap=plt.cm.Reds, vmin=0, vmax=1, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    top_edges = list(zip(pairs_df["gene_a"].head(10), pairs_df["gene_b"].head(10)))
    nx.draw_networkx_edges(G, pos, edgelist=top_edges, width=3, edge_color="darkred", ax=ax)
    ax.set_title("Gene interaction network\n(size = network importance, color = structural risk, red = top predicted pairs)")
    st.subheader("Network visualization")
    st.pyplot(fig)

    progress.progress(100, text="Done.")

    # Downloads
    st.download_button("Download single-variant scores (CSV)", variants_df.to_csv(index=False),
                        file_name="single_variant_scores.csv")
    st.download_button("Download pair interaction scores (CSV)", pairs_df.to_csv(index=False),
                        file_name="pair_interaction_scores.csv")
else:
    st.info("Enter a disease name in the sidebar and click **Run pipeline** to begin.")
