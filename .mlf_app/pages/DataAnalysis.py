import streamlit as st 

st.header("Exploratory Data Analysis")

st.markdown(
    "This page presents our findings during exploratory data analysis (EDA)."
)
st.image("PricePredictionUI/.mlf_app/pages/download.png")
st.text("Some of the features we recognized as crucial were the CPU (processor), power, number of cores in the CPU, the capacity of the hard disk and many more which is why we decided to ask the user for this input when determining the price and nearest neighbors.")

st.image("./.mlf_app/pages/download (2).png")
st.text("Some of the features had too many options, like operating systems. This is why we decided to group them into 5 categories: MacOS, Windows, Linux, DOS and Other.")

st.text("Not all of the features were numerical, hence we had to perform encoding. For most we chose one-hot encoding, but some categorical features had to many options. Most of them were brands of computers, types of graphics cards, RAM and hard disks. For them, we use label encoding")


st.image("./.mlf_app/pages/download (1).png")

st.text("In this plot, each dot represents a product, so a laptop/computer. The x and y-axes are the first two principal components, which capture most of the variance in our original 5-Dimensional feature space. We can see from the plot that each cluster, assigend with KMeans, is represented with a different color. This visualization shows us that clusters are mostly compacted together, but a seperation is evident and PCA is still able to show groupings. Overall, we can say that our selected features yield meaningful and interpretable segments in our data. The PCA projection provides more interpretability and supports the validity of the clustering process. However, PCA is a linear method, so its ability to separate clusters with nonlinear boundaries is limited — which explains why a few clusters appear to blend.")

st.image("./.mlf_app/pages/WhatsApp Image 2025-05-20 at 21.32.57.jpeg")

st.text("In order to find the optimal value of the k parameter we use the elbow method, as shown in the image above.")
