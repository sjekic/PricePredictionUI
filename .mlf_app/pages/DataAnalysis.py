import streamlit as st 

st.header("Exploratory Data Analysis")

st.markdown(
    "This page presents our findings during exploratory data analysis (EDA)."
)
st.image(".mlf_app/pages/download.png")
st.text("Some of the features we recognized as crucial were the CPU (processor), power, number of cores in the CPU, the capacity of the hard disk and many more which is why we decided to ask the user for this input when determining the price and nearest neighbors.")

st.image(".mlf_app/pages/download (2).png")
st.text("Some of the features had too many options, like operating systems. This is why we decided to group them into 5 categories: MacOS, Windows, Linux, DOS and Other.")

st.text("Not all of the features were numerical, hence we had to perform encoding. For most we chose one-hot encoding, but some categorical features had to many options. Most of them were brands of computers, types of graphics cards, RAM and hard disks. For them, we use label encoding")


st.image(".mlf_app/pages/download (1).png")

st.text("In this plot, each dot represents a product, so a laptop/computer. The x and y-axes are the first two principal components, which capture most of the variance in our original 5-Dimensional feature space. We can see from the plot that each cluster, assigend with KMeans, is represented with a different color. This visualization shows us that clusters are mostly compacted together, but a seperation is evident and PCA is still able to show groupings. Overall, we can say that our selected features yield meaningful and interpretable segments in our data. The PCA projection provides more interpretability and supports the validity of the clustering process. However, PCA is a linear method, so its ability to separate clusters with nonlinear boundaries is limited — which explains why a few clusters appear to blend.")
st.text("Cluster 0: budget-friendly laptops with low SSD, RAM, and CPU performance. Likely basic models for simple tasks like browsing and document editing. \n Cluster 1: powerful machines with high RAM, large SSDs, and strong CPUs. Likely designed for demanding workloads such as data processing, multitasking, or creative software — Powerful Workstations. \n Cluster 2: well-balanced laptops with mid-to-high specs across all features. These are likely all-rounders suited for students, professionals, or general-purpose users. \n Cluster 3: models characterized by extremely high SSD capacity and RAM — likely high-storage devices used for tasks involving large files, such as video editing or engineering simulations.\n Cluster 4: the lightest and smallest models, yet offering good performance. Likely ultraportables or premium travel laptops that focus on portability without sacrificing too much power.")
st.image(".mlf_app/pages/WhatsApp Image 2025-05-20 at 21.32.57 (1).jpeg")

st.text("In order to find the optimal value of the k parameter we use the elbow method, as shown in the image above.")

st.image(".mlf_app/pages/WhatsApp Image 2025-05-20 at 23.36.16 (2).jpeg")
st.text("PCA has been used because it is a popular tool when it comes to reducing  the high dimensional data into lower dimensional data while maximising the variance of the data. So, the model tell us where the direction where the data points vary the most. In the figure above we observe clear separation among the clusters. We observe that clusters 0, 1, 2 and 4  are more compacted together, which is positive. While the cluster 3, red, is more spread out, suggesting the potential presence of outliers. We also identify a potential overlap between clusters 1 and 2, orange and green respectively suggesting that they might share some features. ")

st.image(".mlf_app/pages/WhatsApp Image 2025-05-20 at 23.36.16.jpeg")
st.text("tSNE, on the other hand, is a non-linear method for visualising the higher-dimensional data into lower dimensions which focuses on pairwise similarities. For the tSNE we decided to project our clusters against the price, to analyze the differences in the price based on the cluster. In the diagram above, we observe that higher price points are clustered in the middle part of the tSNE. On the other hand, other clusters seem to have similar price ranges. Therefore, we can conclude that the model is able to indirectly capture pricing differences among clusters.  ")

st.image(".mlf_app/pages/WhatsApp Image 2025-05-20 at 23.36.16 (1).jpeg")
st.text("Finally, UMAP, preserves both local and global structure. In the figure above, we observe that clusters 1 and 2 are slightly overlapping, confirming what we identified earlier: clusters 1 and 2 might be sharing some features, but they remain fairly separated and preserve distinction. Clusters 0 and 4, blue and purple, are overlapping at the center of the representation suggesting similarities in features. Cluster 3 is very different from others and compact which might suggest a high end group. Overall, UMAP supports the validity of the clusters that were created, yet we identified some potential overlapping.")
