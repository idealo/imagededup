import os
import shutil
import pandas as pd
import streamlit as st
from PIL import Image
from pprint import pprint
from imagededup.methods import PHash
from imagededup.utils import plot_duplicates

st.set_page_config(
    page_title='Imagededup Webapp',
    page_icon='🖼',
    layout='centered',
    initial_sidebar_state='auto',
)


@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def clean_directory(dir):
    for filename in os.listdir(dir):
        filepath = os.path.abspath(os.path.join(dir, filename))
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def create_dataframe():
    df = pd.DataFrame(columns=['duplicate_images'])
    return df

@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def find_duplicate_imgs():
    phasher = PHash()
    encodings = phasher.encode_images(image_dir='uploads/')
    duplicates = phasher.find_duplicates(encoding_map=encodings)
    dup_imgs = []
    [dup_imgs.append(x) for x in list(duplicates.values()) if len(x)!=0]
    final_dup_imgs = [j for i in dup_imgs for j in i]
    return final_dup_imgs


if __name__ == '__main__':
    clean_directory('uploads/')

    main_image = Image.open('static/main_banner.png')
    st.image(main_image,use_column_width='auto')
    st.title('✨ Image Deduplicator 🏜')
    st.info(' Let me help you find exact and near duplicates in an image collection 😉')

    uploaded_files = st.file_uploader('Upload Images 🚀', type=['png', 'jpg', 'bmp', 'jpeg'], accept_multiple_files=True)
    with st.spinner(f'Finding duplicates... This may take several minutes depending on the number of images uploaded 💫'):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with open(os.path.join('uploads/', uploaded_file.name), 'wb') as f:
                    f.write((uploaded_file).getbuffer())

            final_dup_imgs = find_duplicate_imgs()

            df = create_dataframe()
            df['duplicate_images'] = final_dup_imgs
            downloadable_csv = convert_df(df)

            st.dataframe(df)

            st.download_button(
               'Download as CSV 📝',
               downloadable_csv,
               'list of duplicate images.csv',
               'text/csv',
               key='download-csv'
            )
        else:
            st.warning('⚠ Please upload your images! 😯')



    st.markdown("<br><hr><center>Made with ❤️ by <a href='mailto:ralhanprateek@gmail.com?subject=imagededup WebApp!&body=Please specify the issue you are facing with the app.'><strong>Prateek Ralhan</strong></a> with the help of [imagededup](https://github.com/idealo/imagededup) built by [idealo](https://github.com/idealo) ✨</center><hr>", unsafe_allow_html=True)
    st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)


