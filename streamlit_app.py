import streamlit as st
from veriexcite import (
    extract_bibliography_section,
    split_references,
    search_title,
    set_google_api_key,
    ReferenceStatus,  # new import
)
import io
import pandas as pd
import PyPDF2


def extract_text_from_pdf(pdf_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """Validates if the file is a PDF, then extract text."""
    if not pdf_file.name.lower().endswith(".pdf"):
        raise ValueError("Uploaded file is not a PDF.")
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    pdf_content = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            pdf_content += page_text + "\n"
    return pdf_content


def process_and_verify(bib_text: str, keywords=["Reference", "Bibliography", "Works Cited"]) -> pd.DataFrame:
    """Extracts, processes, and verifies references."""
    # Create containers in the main area
    progress_text = st.empty()
    placeholder = st.empty()
    progress_text.text("正在提取参考文献...")

    try:
        references = split_references(bib_text)
    except ValueError as e:
        st.error(str(e))
        return pd.DataFrame()

    ref_type_dict = {"journal_article": "期刊文章", "preprint": "预印本", "conference_paper": "会议论文",
                     "book": "书籍", "book_chapter": "书籍章节", "non_academic_website": "网站"}
    status_emoji = {
        "validated": "✅已验证",
        "invalid": "❌无效",
        "not_found": "⚠️未找到",
        "Pending": "⏳处理中"
    }

    results = []
    for idx, ref in enumerate(references):
        results.append({
            "Index": idx,
            "第一作者": ref.author,
            "年份": str(ref.year),
            "标题": ref.title,
            "类型": ref_type_dict.get(ref.type, ref.type),
            "DOI": ref.DOI,
            "URL": ref.URL,
            "原始文本": ref.bib,
            "状态": "处理中",
            "说明": "处理中"
        })

    df = pd.DataFrame(results)

    # if URL is empty, and DOI is not empty: if DOI start wih https://, fill url with doi. Else, fill url with doi.org link
    df['URL'] = df.apply(
        lambda x: x['DOI'] if pd.notna(x['DOI']) and x['DOI'] != '' and (pd.isna(x['URL']) or x['URL'] == '') and x[
            'DOI'].startswith('https://') else f'https://doi.org/{x["DOI"]}' if pd.notna(x['DOI']) and x[
            'DOI'] != '' and (pd.isna(x['URL']) or x['URL'] == '') else x['URL'], axis=1)

    column_config = {
        "第一作者": st.column_config.TextColumn(
            help="第一作者的姓氏或机构"),
        "年份": st.column_config.TextColumn(width="small"),
        "URL": st.column_config.LinkColumn(width="medium"),
        "原始文本": st.column_config.TextColumn(
            "原始参考文献文本",  # Display name
            help="悬停查看完整文本",  # Tooltip message
            width="medium",  # Width of the column
        ),
        "状态": st.column_config.TextColumn(
            help="参考文献验证状态"
        ),
        "说明": st.column_config.TextColumn(
            help="验证结果的说明"
        )
    }

    df_display = df[[
        '第一作者', '年份', '标题', '类型', 'URL', '原始文本', '状态', '说明']]
    placeholder.dataframe(df_display, use_container_width=True, column_config=column_config)

    verified_count = 0
    warning_count = 0
    progress_text.text(f"已验证: {verified_count} | 无效/未找到: {warning_count}")

    for index, row in df.iterrows():
        result = search_title(references[index])
        df.loc[index, "状态"] = status_emoji.get(result.status.value, result.status.value)
        df.loc[index, "说明"] = result.explanation
        if result.status == ReferenceStatus.VALIDATED:
            verified_count += 1
        else:
            warning_count += 1
        df_display = df[[
            '第一作者', '年份', '标题', '类型', 'URL', '原始文本', '状态', '说明']]
        placeholder.dataframe(df_display, use_container_width=True, column_config=column_config)
        progress_text.text(f"已验证: {verified_count} | 无效/未找到: {warning_count}")

    return df


def main():
    st.set_page_config(page_title="学佑星途论文引用检测工具", page_icon="🔍", layout="wide", initial_sidebar_state="expanded",
                       menu_items={
                           "About": "这是一个用于验证学术论文中引用文献的工具。在 [GitHub](https://github.com/ykangw/VeriExCiting) 查看源代码。"})

    st.title("学佑星途论文引用检测工具")
    st.write(
        "此工具帮助验证学术论文（PDF格式）中引用文献的存在性。 "
        "它提取参考文献，解析引用，并检查其有效性。"
    )

    with st.sidebar:
        st.header("输入")
        pdf_files = st.file_uploader("上传一个或多个PDF文件", type="pdf", accept_multiple_files=True)

        st.write(
            "您可以在 [Google AI Studio](https://ai.google.dev/aistudio) 申请 Gemini API 密钥，每天免费提供1500次请求。")
        api_key = st.text_input("输入您的Google Gemini API密钥:", type="password")

    if st.sidebar.button("开始验证"):
        if not pdf_files:
            st.warning("请至少上传一个PDF文件。")
            return

        if not api_key:
            st.warning("请输入Google Gemini API密钥。")
            return

        try:
            set_google_api_key(api_key)
            all_results = []

            for pdf_file in pdf_files:
                subheader = st.subheader(f"正在处理: {pdf_file.name}")
                bib_text = extract_bibliography_section(extract_text_from_pdf(pdf_file))

                # Display extracted bibliography text with expander
                with st.expander(f"{pdf_file.name} 的提取参考文献文本"):
                    st.text_area("提取的文本", bib_text, height=200, label_visibility="hidden")

                results_df = process_and_verify(bib_text)
                results_df['源文件'] = pdf_file.name
                all_results.append(results_df)
                subheader.subheader(f"已完成: {pdf_file.name}")

            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                csv = combined_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="下载所有结果为CSV",
                    data=csv,
                    file_name='学佑星途引用检测结果.csv',
                    mime='text/csv',
                )

        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"发生错误: {e}")


if __name__ == "__main__":
    main()