import streamlit as st
import os
import time

def load_css(file_name):
    # Get file modification time for cache busting
    try:
        mtime = os.path.getmtime(file_name)
        cache_buster = int(mtime)
    except:
        cache_buster = int(time.time())
    
    with open(file_name) as f:
        css_content = f.read()
        # Add cache buster comment to force reload
        st.markdown(f'<style>/* Cache: {cache_buster} */\n{css_content}</style>', unsafe_allow_html=True)

def card_header(title = None, subtitle = None, desc = None):

    header_container_start = "<div class='card-header'>"
    header_content = ""
    header_container_end = "</div>"

    if title:
        header_content += f'''<h1 class="card-title">{title}</h1>'''
    if subtitle:
        header_content += f'''<h2 class="card-subtitle">{subtitle}</h2>'''
    if desc:
        header_content += f'''<p class="card-desc">{desc}</p>'''

    st.markdown(
        f'''
            {header_container_start}
            {header_content}
            {header_container_end}
        ''',
        unsafe_allow_html=True
    )
