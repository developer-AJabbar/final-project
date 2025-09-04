import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import time
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
from pyvis.network import Network
from streamlit.components.v1 import html as components_html
import base64
import os
import tempfile

def encode_units(x):
    if x <= 0:
        return False
    if x >= 1:
        return True

# Set page configuration
st.set_page_config(
    page_title="🌟 Apriori Rule Explorer Pro",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'itemsets' not in st.session_state:
    st.session_state.itemsets = None
if 'rules' not in st.session_state:
    st.session_state.rules = None
if 'network_file' not in st.session_state:
    st.session_state.network_file = None
if 'min_support' not in st.session_state:
    st.session_state.min_support = 0.05
if 'min_confidence' not in st.session_state:
    st.session_state.min_confidence = 0.5
if 'min_lift' not in st.session_state:
    st.session_state.min_lift = 0.5

# Consolidated CSS with contact section styles merged
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&family=Poppins:wght@300;400;600;700&display=swap');
    
    :root {
        --primary: #4DD0E1;
        --secondary: #9575CD;
        --dark: #2D3748;
        --light: #F8F9FA;
        --success: #26A69A;
        --warning: #FFB74D;
        --danger: #EF5350;
    }
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
    }
    
    .main-container {
        background-color: rgba(255, 255, 255, 0.98);
        padding: 3rem;
        border-radius: 25px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        margin: 2rem auto;
        max-width: 1300px;
        animation: fadeIn 1s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .main-container::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(77,208,225,0.1) 0%, rgba(149,117,205,0.05) 50%, transparent 70%);
        animation: rotate 20s linear infinite;
        z-index: -1;
    }
    
    .header-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 700;
        color: var(--dark);
        margin-bottom: 1.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        font-family: 'Montserrat', sans-serif;
        position: relative;
        padding-bottom: 1rem;
    }
    
    .header-title::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 150px;
        height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        border-radius: 2px;
    }
    
    .graph-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
    }
    
    .interactive-card {
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .interactive-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Contact section styling */
    .contact-section {
        background: linear-gradient(135deg, #4DD0E1 0%, #9575CD 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 3rem;
        color: white;
        box-shadow: 0 10px 30px rgba(77, 208, 225, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .contact-section::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
        z-index: 0;
    }
    
    .contact-title {
        font-size: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }
    
    .contact-grid {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1.5rem;
        position: relative;
        z-index: 1;
    }
    
    .contact-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
        flex: 1;
        min-width: 200px;
        max-width: 250px;
    }
    
    .contact-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.25);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }
    
    .contact-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: white;
    }
    
    .contact-link {
        color: white;
        text-decoration: none;
        font-weight: 600;
        display: block;
        margin-top: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .contact-link:hover {
        color: #F8F9FA;
        text-decoration: underline;
    }
    
    .contact-text {
        margin-top: 1.5rem;
        text-align: center;
        font-style: italic;
        position: relative;
        z-index: 1;
    }
    
    @media (max-width: 768px) {
        .main-container {
            padding: 1.5rem;
            border-radius: 15px;
        }
        
        .header-title {
            font-size: 2.2rem;
        }
        
        .subheader {
            font-size: 1.3rem;
        }
        
        .col1, .col2 {
            width: 100% !important;
            padding: 0 !important;
        }
        
        .contact-grid {
            flex-direction: column;
            align-items: center;
        }
        
        .contact-card {
            width: 100%;
            max-width: none;
        }
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

# Enhanced header with particles effect
st.markdown("""
    <div style="position: relative; overflow: hidden; border-radius: 20px; margin-bottom: 2rem;">
        <canvas id="particles-js" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;"></canvas>
        <div class="header-title">🔍 Apriori Association Rule Explorer Pro</div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
    <script>
    particlesJS("particles-js", {
        "particles": {
            "number": {"value": 80, "density": {"enable": true, "value_area": 800}},
            "color": {"value": "#4DD0E1"},
            "shape": {"type": "circle", "stroke": {"width": 0, "color": "#000000"}, "polygon": {"nb_sides": 5}},
            "opacity": {"value": 0.5, "random": false, "anim": {"enable": false, "speed": 1, "opacity_min": 0.1, "sync": false}},
            "size": {"value": 3, "random": true, "anim": {"enable": false, "speed": 40, "size_min": 0.1, "sync": false}},
            "line_linked": {"enable": true, "distance": 150, "color": "#9575CD", "opacity": 0.4, "width": 1},
            "move": {"enable": true, "speed": 2, "direction": "none", "random": false, "straight": false, "out_mode": "out", "bounce": false, "attract": {"enable": false, "rotateX": 600, "rotateY": 1200}}
        },
        "interactivity": {
            "detect_on": "canvas",
            "events": {
                "onhover": {"enable": true, "mode": "grab"},
                "onclick": {"enable": true, "mode": "push"},
                "resize": true
            },
            "modes": {
                "grab": {"distance": 140, "line_linked": {"opacity": 1}},
                "bubble": {"distance": 400, "size": 40, "duration": 2, "opacity": 8, "speed": 3},
                "repulse": {"distance": 200, "duration": 0.4},
                "push": {"particles_nb": 4},
                "remove": {"particles_nb": 2}
            }
        },
        "retina_detect": true
    });
    </script>
""", unsafe_allow_html=True)

# Main container
st.markdown("""<div class='main-container'>""", unsafe_allow_html=True)

# Info box with animation
st.markdown("""
    <div class='info-box' style="animation: fadeIn 0.8s ease-in-out;">
        <h3 style="margin-top: 0; color: inherit;">📊 Advanced Market Basket Analysis</h3>
        <p>Upload your transactional dataset to discover hidden relationships between products using the Apriori algorithm. This enhanced version includes:</p>
        <ul>
            <li>Interactive network visualizations</li>
            <li>Advanced filtering options</li>
            <li>Beautiful visualizations</li>
            <li>Export capabilities</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# File upload section with enhanced UI
st.markdown('<div class="subheader" style="animation: fadeIn 0.8s ease-in-out;"></div>', unsafe_allow_html=True)

with st.expander("Dataset Requirements", expanded=False):
    st.markdown("""
        - CSV format with 'Member_number' and 'itemDescription' columns
        - Items should be comma-separated in 'itemDescription'
        - First row should contain headers
        - File size limit: 200MB
    """)

file = st.file_uploader("📁 Upload Your Dataset", type=["csv"], key="file_uploader",
                       help="Upload your transaction data file")

if file:
    st.session_state.df = pd.read_csv(file)
    
    # Dataset preview with tabs
    st.markdown('<div class="subheader">🔍 Dataset Explorer Top 20 Most Frequent Item </div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Preview", "Statistics", "Missing Values"])
    
    with tab1:
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
    
    with tab2:
        st.write(st.session_state.df.describe())
    
    with tab3:
        st.write(st.session_state.df.isnull().sum())
    
    # Enhanced frequent items visualization
    st.markdown('<div class="subheader">📊 Product Frequency Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        all_items = []
        for items in st.session_state.df['itemDescription']:
            if pd.notna(items):
                all_items.extend([item.strip() for item in str(items).split(',') if item.strip()])
        item_counts = pd.Series(all_items).value_counts().sort_values(ascending=False)[:20]
        
        # Interactive plot with Plotly
        fig = px.bar(item_counts, 
                    x=item_counts.index, 
                    y=item_counts.values,
                    color=item_counts.values,
                    color_continuous_scale='Viridis',
                    labels={'x':'Products', 'y':'Frequency'},
                    title='Top 20 Most Frequent Products')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Product Distribution")
        fig = px.pie(names=item_counts.index[:10], 
                    values=item_counts.values[:10],
                    hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
        
    # Apriori parameters with enhanced UI
    st.markdown('<div class="subheader">⚙️ Algorithm Parameters</div>', unsafe_allow_html=True)
    
    with st.form("parameters_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.min_support = st.slider("Minimum Support", 0.01, 1.0, st.session_state.min_support, 0.01,
                                   help="Minimum frequency of itemset in the dataset")
        
        with col2:
            st.session_state.min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, st.session_state.min_confidence, 0.05,
                                     help="Minimum confidence for association rules")
        
        with col3:
            st.session_state.min_lift = st.slider("Minimum Lift", 1.0, 10.0, st.session_state.min_lift, 0.1,
                                help="Minimum lift value for meaningful rules")
        
        submitted = st.form_submit_button("🚀 Run Advanced Analysis", 
                                         use_container_width=True,
                                         help="Generate association rules with current parameters")

    if submitted:
        with st.spinner("🔍 Mining advanced association rules..."):
            progress_bar = st.progress(0)
            
            # Processing steps with progress updates
            transaction_data = []
            for _, row in st.session_state.df.iterrows():
                member = row['Member_number']
                items = [i.strip() for i in str(row['itemDescription']).split(',') if i.strip()]
                transaction_data.extend([{"Member_number": member, "item": item} for item in items])
            
            progress_bar.progress(25)
            processed_df = pd.DataFrame(transaction_data)
            basket = processed_df.pivot_table(index='Member_number', columns='item', 
                                            aggfunc=lambda x: 1, fill_value=0)
            
            progress_bar.progress(50)
            st.session_state.itemsets = apriori(basket, min_support=st.session_state.min_support, use_colnames=True)
            
            progress_bar.progress(75)
            st.session_state.rules = association_rules(st.session_state.itemsets, metric="lift", 
                                    min_threshold=st.session_state.min_lift)
            
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()

if st.session_state.itemsets is not None and not st.session_state.itemsets.empty:
    # Enhanced results display
    st.markdown('<div class="subheader">📦 Frequent Itemsets</div>', unsafe_allow_html=True)
    
    # Interactive itemset explorer
    st.session_state.itemsets['length'] = st.session_state.itemsets['itemsets'].apply(lambda x: len(x))
    length_filter = st.slider("Filter by itemset length", 
                             int(st.session_state.itemsets['length'].min()), 
                             int(st.session_state.itemsets['length'].max()),
                             (1, 3))
    
    filtered_itemsets = st.session_state.itemsets[(st.session_state.itemsets['length'] >= length_filter[0]) & 
                               (st.session_state.itemsets['length'] <= length_filter[1])]
    st.dataframe(filtered_itemsets.sort_values('support', ascending=False), 
                use_container_width=True)

if st.session_state.rules is not None and not st.session_state.rules.empty:
    # Enhanced rules visualization
    st.markdown('<div class="subheader">🔗 Association Rules</div>', unsafe_allow_html=True)
    
    # Interactive rules filtering
    col1, col2 = st.columns(2)
    with col1:
        confidence_filter = st.slider("Filter by confidence", 
                                   float(st.session_state.rules['confidence'].min()), 
                                   float(st.session_state.rules['confidence'].max()),
                                   (0.5, 1.0))
    
    with col2:
        lift_filter = st.slider("Filter by lift", 
                              float(st.session_state.rules['lift'].min()), 
                              float(st.session_state.rules['lift'].max()),
                              (st.session_state.min_lift, float(st.session_state.rules['lift'].max())))
    
    filtered_rules = st.session_state.rules[(st.session_state.rules['confidence'] >= confidence_filter[0]) & 
                             (st.session_state.rules['confidence'] <= confidence_filter[1]) &
                             (st.session_state.rules['lift'] >= lift_filter[0]) & 
                             (st.session_state.rules['lift'] <= lift_filter[1])]
    
    st.dataframe(filtered_rules.style.format({
        "support": "{:.4f}",
        "confidence": "{:.4f}",
        "lift": "{:.4f}"
    }).background_gradient(cmap='Blues'), 
    use_container_width=True)

    # Enhanced network graph
    st.markdown('<div class="subheader">🌐 Interactive Rules Network</div>', unsafe_allow_html=True)
    
    rules_filtered = filtered_rules.head(30)
    if not rules_filtered.empty:
        G = nx.DiGraph()
        for index, row in rules_filtered.iterrows():
            antecedent_str = ', '.join(list(row['antecedents']))
            consequent_str = ', '.join(list(row['consequents']))
            G.add_edge(antecedent_str, consequent_str, 
                      weight=row['lift'], 
                      confidence=row['confidence'])
        
        # Create interactive plot with pyvis
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(G)
        
        # Customize node appearance
        for node in net.nodes:
            node['size'] = 15 + G.degree(node['id']) * 2
            node['color'] = '#4DD0E1' if G.in_degree(node['id']) == 0 else '#9575CD'
        
        # Save to a temporary file
        temp_dir = tempfile.mkdtemp()
        network_path = os.path.join(temp_dir, "network.html")
        net.save_graph(network_path)
        st.session_state.network_file = network_path
        
        # Display the network
        HtmlFile = open(network_path, 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components_html(source_code, height=800, width=1000)
    else:
        st.warning("No rules meet the current filtering criteria")

    # Download options
    st.markdown("### 📤 Export Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = filtered_itemsets.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="frequent_itemsets.csv">Download Itemsets (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        csv = filtered_rules.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="association_rules.csv">Download Rules (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    with col3:
        if st.session_state.network_file:
            with open(st.session_state.network_file, "rb") as f:
                bytes_data = f.read()
                b64 = base64.b64encode(bytes_data).decode()
                href = f'<a href="data:file/html;base64,{b64}" download="rules_network.html">Download Network (HTML)</a>'
                st.markdown(href, unsafe_allow_html=True)



# Close main container
st.markdown("</div>", unsafe_allow_html=True)
