import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import gensim.downloader as api
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import OPTICS

def display_intro() -> None:
    """Display a brief introduction to the app."""
    st.title('Word Embeddings Visualization 📊')
    st.write("""
             In this application, we utilize embeddings from the Google News word2vec model with 300 dimensions 
             to demonstrate dimensionality reduction using PCA and t-SNE, as well as data clustering with the OPTICS algorithm. 
             Dimensionality reduction is an important technique for visualizing high-dimensional data in a two or 
             three-dimensional space. Similarly, OPTICS clustering helps to identify the structure of data points 
             in this reduced dimensional space, revealing patterns and groupings that can be crucial for understanding 
             the natural groupings within the data.
            """)

@st.cache_data()
def load_model() -> object:
    """Load the word2vec model from gensim-data."""
    return api.load("word2vec-google-news-300")

def get_word_vectors(model, word_list) -> tuple:
    """
    Get the word vectors for the given list of words.
    
    Args:
        model (object): The word2vec model.
        word_list (list): The list of words.
    
    Returns:
        tuple: The word vectors and the filtered word list.
    """
    word_vecs = []
    filtered_word_list = []
    for word in word_list:
        if word in model:
            word_vecs.append(model[word])
            filtered_word_list.append(word)
    return word_vecs, filtered_word_list

def plot_pca(reduced_vecs, filtered_word_list, num_pca_components) -> object:
    """
    Plot the PCA visualization using Plotly.
    
    Args:
        reduced_vecs (numpy.ndarray): The reduced vectors.
        filtered_word_list (list): The filtered word list.
        num_pca_components (int): The number of PCA components (2 or 3).
    
    Returns:
        object: The PCA visualization (Plotly figure).
    """
    fig_pca = None

    if num_pca_components == 2:
        fig_pca = px.scatter(x=reduced_vecs[:, 0], y=reduced_vecs[:, 1], text=filtered_word_list, 
                             labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}, 
                             title='PCA visualization of word embeddings',
                             width=800, height=600)
        fig_pca.update_traces(marker=dict(color='orange', opacity=0.6, size=12),
                              selector=dict(mode='markers+text'))
        fig_pca.update_layout(paper_bgcolor='rgba(0,0,0,0)', 
                      plot_bgcolor='rgba(0,0,0,0)', 
                      font=dict(family='Arial, sans-serif', size=14),
                      margin=dict(l=40, r=40, b=40, t=40))  

        for trace in fig_pca.data:
            trace.textposition = 'top center'
            trace.textfont.size = 10
        
    elif num_pca_components == 3:
        fig_pca = go.Figure(data=[go.Scatter3d(x=reduced_vecs[:, 0], y=reduced_vecs[:, 1], z=reduced_vecs[:, 2], 
                                               mode='markers+text', text=filtered_word_list,
                                               marker=dict(color='orange', size=12, opacity=0.6))])
        fig_pca.update_layout(scene=dict(xaxis_title='PCA Component 1',
                                         yaxis_title='PCA Component 2',
                                         zaxis_title='PCA Component 3'),
                              width=800, height=600,
                              title='PCA visualization of word embeddings')
        for trace in fig_pca.data:
            trace.textposition = 'top center'
            trace.textfont.size = 10
        
    return fig_pca

def plot_tsne(embedding_2d, filtered_word_list, text_offset=0.1) -> object:
    """
    Plot the t-SNE visualization using Plotly with improved visual style.
    
    Args:
        embedding_2d (numpy.ndarray): The 2D embedding.
        filtered_word_list (list): The filtered word list.
        text_offset (float): Vertical offset for text labels.
    
    Returns:
        object: The t-SNE visualization.
    """
    fig = px.scatter(x=embedding_2d[:, 0], y=embedding_2d[:, 1], text=filtered_word_list,
                     labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
                     title='t-SNE visualization of word embeddings',
                     opacity=0.7,
                     width=800, height=600)
    
    fig.update_traces(marker=dict(color='blue', opacity=0.6, size=14),
                              selector=dict(mode='markers+text'))
    
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', 
                      plot_bgcolor='rgba(0,0,0,0)',  
                      font=dict(family='Arial, sans-serif', size=12),
                      margin=dict(l=40, r=40, b=40, t=40)) 
    
    for trace in fig.data:
        trace.textposition = 'top center'
        trace.textfont.size = 10
    
    return fig

def perform_optics_clustering(embeddings, min_samples=2):
    """
    Perform OPTICS clustering on the embeddings.
    
    Args:
        embeddings (numpy.ndarray): The word embeddings.
    
    Returns:
        numpy.ndarray: The cluster labels.
    """
    optics_model = OPTICS(min_samples=min_samples, xi=0.05, min_cluster_size=0.1)
    labels = optics_model.fit_predict(embeddings)
    return labels

def plot_optics(embedding_2d, labels, filtered_word_list):
    """
    Plot the results of OPTICS clustering using Plotly.
    
    Args:
        embedding_2d (numpy.ndarray): The 2D embeddings (after PCA/t-SNE).
        labels (numpy.ndarray): The cluster labels from OPTICS.
        filtered_word_list (list): The filtered list of words.
        
    Returns:
        object: The Plotly figure for OPTICS clustering visualization.
    """
    fig = px.scatter(x=embedding_2d[:, 0], y=embedding_2d[:, 1], text=filtered_word_list,
                     color=labels, labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
                     title='OPTICS Clustering of Word Embeddings', width=800, height=600)
    
    fig.update_traces(marker=dict(opacity=0.8, size=10),
                      selector=dict(mode='markers+text'))
    
    fig.update_layout(margin=dict(l=40, r=40, b=40, t=40),
                      font=dict(family='Arial, sans-serif', size=12))
    
    return fig

def main():
    """Main function of the app."""
    display_intro()

    model = load_model()

    # Inicialization of the session state variables
    if 'show_pca' not in st.session_state:
        st.session_state['show_pca'] = False
    if 'show_tsne' not in st.session_state:
        st.session_state['show_tsne'] = False
    if 'show_optics' not in st.session_state:
        st.session_state['show_optics'] = False

    # Input the words
    words = st.text_input("Enter words (separated by commas): 📝")

    if words:
        word_list = [word.strip() for word in words.split(",")]
        word_vecs, filtered_word_list = get_word_vectors(model, word_list)

        if word_vecs:
            word_vecs_array = np.array(word_vecs)
            
            # Verify if there are enough words to perform PCA
            max_components = min(len(filtered_word_list), word_vecs_array.shape[1])
            if max_components > 2:
                num_pca_components = st.slider("Number of components for PCA:", 2, max_components, 2)
            else:
                st.warning("There are not enough components to perform PCA.")
                return
            
            # Define the max and min values for perplexity
            min_perplexity = 2
            max_perplexity = max(5, len(filtered_word_list) - 1)

            # Certify that there are enough words to adjust the perplexity
            if max_perplexity > min_perplexity:
                default_perplexity = min(30, max_perplexity)
                perplexity_value = st.slider("Perplexity for t-SNE:", min_perplexity, max_perplexity, default_perplexity)
            else:
                st.warning("There are not enough words to adjust the t-SNE perplexity.")
                return
    
            # Processing and plotting
            if st.button("Process"):
                st.session_state['show_pca'] = True
                st.session_state['show_tsne'] = True
                st.session_state['show_optics'] = False

            if st.session_state['show_pca']:
                pca = PCA(n_components=num_pca_components)
                reduced_vecs = pca.fit_transform(word_vecs_array)
                st.info(f"Explained variance by {num_pca_components} components: {sum(pca.explained_variance_ratio_):.2%}")
                if num_pca_components == 2 or num_pca_components == 3:
                    fig_pca = plot_pca(reduced_vecs, filtered_word_list, num_pca_components)
                    st.plotly_chart(fig_pca)
    
            if st.session_state['show_tsne']:
                tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
                embedding_2d = tsne.fit_transform(reduced_vecs)
                fig_tsne = plot_tsne(embedding_2d, filtered_word_list)
                st.plotly_chart(fig_tsne)

            if st.session_state['show_tsne'] and st.session_state['show_pca']:
                max_samples = len(word_vecs_array)
                min_samples = st.slider('Min Samples for OPTICS', 2, max(max_samples // 2, 2), 2)
                if st.button("Cluster with OPTICS"):
                    st.session_state['show_optics'] = True

            if st.session_state['show_optics']:
                with st.spinner('Performing OPTICS clustering...'):
                    labels = perform_optics_clustering(embedding_2d, min_samples)
                    fig_optics = plot_optics(embedding_2d, labels, filtered_word_list)
                    st.plotly_chart(fig_optics)
        else:
            st.warning("⚠️ None of the entered words are in the model's vocabulary.")
    else:
        st.write("Please enter some words to begin.")

if __name__ == '__main__':
    main()