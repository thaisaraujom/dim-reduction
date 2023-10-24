import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import gensim.downloader as api
import numpy as np

def display_intro() -> None:
    """Display a brief introduction to the app."""
    st.title('Word Embeddings Visualization 📊')
    st.write("""
             In this application, we utilize embeddings from the Google News word2vec model with 300 dimensions 
             to demonstrate dimensionality reduction 
             using PCA and t-SNE. Dimensionality reduction is an 
             important technique for visualizing high-dimensional data in a two or 
             three-dimensional space.
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
    Plot the PCA visualization.
    
    Args:
        reduced_vecs (numpy.ndarray): The reduced vectors.
        filtered_word_list (list): The filtered word list.
        num_pca_components (int): The number of PCA components.
    
    Returns:
        object: The PCA visualization.
    """
    plt.style.use('seaborn-whitegrid')
    fig_pca = None

    if num_pca_components == 2:
        fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
        ax_pca.scatter(reduced_vecs[:, 0], reduced_vecs[:, 1], color='orange', alpha=0.6)
        for i, word in enumerate(filtered_word_list):
            ax_pca.annotate(word, (reduced_vecs[i, 0], reduced_vecs[i, 1]), fontsize=10, ha='right')
        ax_pca.spines["top"].set_visible(False)
        ax_pca.spines["right"].set_visible(False)
        ax_pca.grid(color='lightgray', linestyle='--', linewidth=0.5)
        fig_pca.suptitle('PCA visualization of word embeddings', fontsize=14, fontweight='bold')

    elif num_pca_components == 3:
        fig_pca = plt.figure(figsize=(10, 6))
        ax_pca = fig_pca.add_subplot(111, projection='3d')
        ax_pca.scatter(reduced_vecs[:, 0], reduced_vecs[:, 1], reduced_vecs[:, 2], color='orange', alpha=0.6, s=50)
        for i, word in enumerate(filtered_word_list):
            ax_pca.text(reduced_vecs[i, 0], reduced_vecs[i, 1], reduced_vecs[i, 2], word, fontsize=10, ha='right')

        # Define the background color of the plot to white
        ax_pca.set_facecolor('white')

        # Colors and styles
        ax_pca.w_xaxis.pane.set_edgecolor('lightgrey')
        ax_pca.w_yaxis.pane.set_edgecolor('lightgrey')
        ax_pca.w_zaxis.pane.set_edgecolor('lightgrey')
        ax_pca.w_xaxis.line.set_color('lightgrey')
        ax_pca.w_yaxis.line.set_color('lightgrey')
        ax_pca.w_zaxis.line.set_color('lightgrey')

        # Grid lines
        ax_pca.xaxis._axinfo["grid"]['color'] =  (0.8, 0.8, 0.8, 0.3)
        ax_pca.yaxis._axinfo["grid"]['color'] =  (0.8, 0.8, 0.8, 0.3)
        ax_pca.zaxis._axinfo["grid"]['color'] =  (0.8, 0.8, 0.8, 0.3)

        # Define the color of the axis labels
        ax_pca.tick_params(axis='both', which='major', colors='grey')
        fig_pca.suptitle('PCA visualization of word embeddings', fontsize=14, fontweight='bold')

    return fig_pca

def plot_tsne(embedding_2d, filtered_word_list) -> object:
    """
    Plot the t-SNE visualization.
    
    Args:
        embedding_2d (numpy.ndarray): The 2D embedding.
        filtered_word_list (list): The filtered word list.
    
    Returns:
        object: The t-SNE visualization.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], color='blue', alpha=0.6)
    for i, word in enumerate(filtered_word_list):
        ax.annotate(word, (embedding_2d[i, 0], embedding_2d[i, 1]), fontsize=10, ha='right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5)
    fig.suptitle('t-SNE visualization of word embeddings', fontsize=14, fontweight='bold')
    return fig

def main():
    """Main function of the app."""
    display_intro()

    model = load_model()

    # Input the words
    words = st.text_input("Enter words (separated by commas): 📝")

    # Verify if the user has entered any words
    if words:  
        word_list = [word.strip() for word in words.split(",")]
        word_vecs, filtered_word_list = get_word_vectors(model, word_list)

        if word_vecs:
            word_vecs_array = np.array(word_vecs)
            
            # Verify if we have enough words to perform PCA
            max_components = min(len(filtered_word_list), word_vecs_array.shape[1])
            if max_components > 2:
                num_pca_components = st.slider("Number of components for PCA:", 2, max_components, 2)
            else:
                st.warning("There are not enough components to perform PCA.")
                return
            
            # Adittional settings
            st.write("### Advanced settings")
            
            # Define the max and min values for perplexity
            min_perplexity = 2
            max_perplexity = max(5, len(filtered_word_list) - 1)

            # Verify if there are enough words to adjust the perplexity
            if max_perplexity > min_perplexity:
                # Certify if the default value is within the range
                default_perplexity = min(30, max_perplexity)
                perplexity_value = st.slider("Perplexity for t-SNE:", min_perplexity, max_perplexity, default_perplexity)
            else:
                st.warning("There are not enough words to adjust the t-SNE perplexity.")
                return
    
            # Verify if the user wants to show the PCA visualization
            if num_pca_components > 3:
                show_pca = False
            else:
                show_pca = st.checkbox("Show PCA visualization before t-SNE?", True)
        else:
            st.warning("⚠️ None of the entered words are in the model's vocabulary.")

        if st.button("Process"):
            try:
                with st.spinner('Processing...'):
                    if word_vecs:
                        word_vecs_array = np.array(word_vecs)

                        # PCA
                        pca = PCA(n_components=num_pca_components)
                        reduced_vecs = pca.fit_transform(word_vecs_array)
                        st.write(f"Explained variance: {sum(pca.explained_variance_ratio_):.2%}")

                        if show_pca:
                            fig_pca = plot_pca(reduced_vecs, filtered_word_list, num_pca_components)
                            st.pyplot(fig_pca)

                        # t-SNE
                        tsne = TSNE(n_components=2, perplexity=min(perplexity_value, len(filtered_word_list)-1), random_state=42)
                        embedding_2d = tsne.fit_transform(reduced_vecs)
                        fig_tsne = plot_tsne(embedding_2d, filtered_word_list)
                        st.pyplot(fig_tsne)
                    else:
                        st.warning("⚠️ None of the entered words are in the model's vocabulary.")
            except Exception as error_msg:
                st.error(f"🔴 An error occurred: {str(error_msg)}")
    else:
        st.write("Please enter some words to begin.")

if __name__ == '__main__':
    main()