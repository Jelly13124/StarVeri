import streamlit as st
from veriexcite import (
    extract_bibliography_section,
    split_references,
    search_title,
    set_google_api_key,
    ReferenceStatus,
)
import io
import pandas as pd
import PyPDF2


def extract_text_from_pdf(pdf_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """校验是否为 PDF，并抽取全文文本。"""
    if not pdf_file.name.lower().endswith(".pdf"):
        raise ValueError("上传的文件不是 PDF。")
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    pdf_content = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            pdf_content += page_text + "\n"
    return pdf_content


def process_and_verify(bib_text: str, keywords=["参考文献", "References", "Bibliography", "Works Cited"]) -> pd.DataFrame:
    """抽取、解析并校验参考文献。"""
    # 进度与占位容器
    progress_text = st.empty()
    placeholder = st.empty()
    progress_text.text("正在提取参考文献...")

    try:
        references = split_references(bib_text)
    except ValueError as e:
        st.error(str(e))
        return pd.DataFrame()

    ref_type_dict = {
        "journal_article": "期刊论文",
        "preprint": "预印本",
        "conference_paper": "会议论文",
        "book": "书籍",
        "book_chapter": "书籍章节",
        "non_academic_website": "网站",
    }
    status_emoji = {
        "validated": "✅ 已验证",
        "invalid": "❌ 无效",
        "not_found": "⚠️ 未找到",
        "Pending": "⏳ 待验证",
    }

    results = []
    for idx, ref in enumerate(references):
        results.append({
            "序号": idx,
            "第一作者": ref.author,
            "年份": str(ref.year),
            "标题": ref.title,
            "类型": ref_type_dict.get(ref.type, ref.type),
            "DOI": ref.DOI,
            "链接": ref.URL,
            "原始文本": ref.bib,
            "状态": "待验证",
            "说明": "待验证"
        })

    df = pd.DataFrame(results)

    # 若 URL 为空但 DOI 存在：若 DOI 以 https:// 开头，直接用作链接；否则补全为 https://doi.org/<DOI>
    df['链接'] = df.apply(
        lambda x: x['DOI'] if pd.notna(x['DOI']) and x['DOI'] != '' and (pd.isna(x['链接']) or x['链接'] == '') and x['DOI'].startswith('https://')
        else (f'https://doi.org/{x["DOI"]}' if pd.notna(x['DOI']) and x['DOI'] != '' and (pd.isna(x['链接']) or x['链接'] == '') else x['链接']),
        axis=1
    )

    column_config = {
        "第一作者": st.column_config.TextColumn(help="第一作者的姓氏，或机构名称"),
        "年份": st.column_config.TextColumn(width="small"),
        "链接": st.column_config.LinkColumn(width="medium"),
        "原始文本": st.column_config.TextColumn(
            "原始参考文献",
            help="悬停可查看完整文本",
            width="medium",
        ),
        "状态": st.column_config.TextColumn(help="参考文献校验状态"),
        "说明": st.column_config.TextColumn(help="结果说明"),
    }

    df_display = df[['第一作者', '年份', '标题', '类型', '链接', '原始文本', '状态', '说明']]
    placeholder.dataframe(df_display, use_container_width=True, column_config=column_config)

    verified_count = 0
    warning_count = 0
    progress_text.text(f"已验证：{verified_count} | 异常/未找到：{warning_count}")

    for index, row in df.iterrows():
        result = search_title(references[index])
        df.loc[index, "状态"] = status_emoji.get(result.status.value, result.status.value)
        df.loc[index, "说明"] = result.explanation
        if result.status == ReferenceStatus.VALIDATED:
            verified_count += 1
        else:
            warning_count += 1
        df_display = df[['第一作者', '年份', '标题', '类型', '链接', '原始文本', '状态', '说明']]
        placeholder.dataframe(df_display, use_container_width=True, column_config=column_config)
        progress_text.text(f"已验证：{verified_count} | 异常/未找到：{warning_count}")

    return df


def main():
    st.set_page_config(
        page_title="学佑星途论文检查",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("学佑星途论文检查")
    st.write(
        "本工具用于对学术论文（PDF）中的参考文献进行核验：自动提取“参考文献”板块，解析各条目，并通过检索验证其存在性与可信度。"
    )

    with st.sidebar:
        st.header("输入")
        pdf_files = st.file_uploader("上传一个或多个 PDF 文件", type="pdf", accept_multiple_files=True)

        st.write("要使用在线检索，需要提供 Google Gemini 的 API Key（个人密钥）。")
        st.markdown(
            "前往 [Google AI Studio](https://ai.google.dev/aistudio) 申请（目前常见配额为每日免费一定次数）。"
        )
        api_key = st.text_input("请输入你的 Google Gemini API Key：", type="password", help="密钥仅用于本地检索与校验，不会上传至服务器。")

    if st.sidebar.button("开始校验"):
        if not pdf_files:
            st.warning("请至少上传一个 PDF 文件。")
            return

        if not api_key:
            st.warning("请输入 Google Gemini API Key。")
            return

        try:
            set_google_api_key(api_key)
            all_results = []

            for pdf_file in pdf_files:
                subheader = st.subheader(f"处理中：{pdf_file.name}")
                bib_text = extract_bibliography_section(extract_text_from_pdf(pdf_file))

                # 展示抽取到的参考文献文本
                with st.expander(f"参考文献抽取结果：{pdf_file.name}"):
                    st.text_area("抽取文本", bib_text, height=200, label_visibility="hidden")

                results_df = process_and_verify(bib_text)
                results_df['来源文件'] = pdf_file.name
                all_results.append(results_df)
                subheader.subheader(f"完成：{pdf_file.name}")

            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                csv = combined_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="下载所有结果（CSV）",
                    data=csv,
                    file_name='学佑星途_引用校验结果.csv',
                    mime='text/csv',
                )

        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"发生错误：{e}")


if __name__ == "__main__":
    main()
