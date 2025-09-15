import streamlit as st
# 导入更新后的后端函数
from veriexcite import (
    extract_text_from_pdf,
    extract_bibliography_section,
    split_references,
    verify_reference_with_search,
    find_replacement_reference,
    set_google_api_key,
    ReferenceStatus,
    VerificationResult,
    ReferenceExtraction,
)
import io
import pandas as pd
import os

# 使用 veriexcite.py 中更强大的 PDF 文本提取功能
def extract_text_from_uploaded_file(pdf_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """使用后端的 fitz 模块从上传的文件中提取文本。"""
    if not pdf_file.name.lower().endswith(".pdf"):
        raise ValueError("上传的文件不是 PDF 格式。")

    # 获取上传文件的字节流
    pdf_bytes = pdf_file.getvalue()
    
    # 为了让 fitz 能够处理，需要先将字节流写入一个临时文件
    # Streamlit Cloud 环境中，/tmp/ 是一个可写的临时目录
    temp_dir = "/tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir) # 如果目录不存在则创建
        
    temp_pdf_path = os.path.join(temp_dir, pdf_file.name)
    
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)
        
    # 调用 veriexcite.py 中基于 PyMuPDF 的函数
    return extract_text_from_pdf(temp_pdf_path)


def process_and_verify(bib_text: str) -> pd.DataFrame:
    """使用新的智能后端来提取、解析并校验参考文献。"""
    progress_text = st.empty()
    placeholder = st.empty()
    progress_text.text("正在从参考文献文本中解析条目...")

    try:
        references: List[ReferenceExtraction] = split_references(bib_text)
    except Exception as e:
        st.error(f"解析参考文献失败：{e}")
        return pd.DataFrame()

    status_emoji = {
        "validated": "✅ 已验证",
        "not_found": "⚠️ 未找到",
    }

    # 准备用于显示的 DataFrame 结构
    results = [
        {
            "作者": ref.author,
            "年份": str(ref.year),
            "标题": ref.title,
            "原始文本": ref.bib,
            "状态": "⏳ 验证中...",
            "说明": "待处理",
            "链接": "",
            "替换建议": "",
        }
        for ref in references
    ]
    df = pd.DataFrame(results)

    column_config = {
        "作者": st.column_config.TextColumn("第一作者", help="第一作者的姓氏或机构名称。"),
        "年份": st.column_config.TextColumn(width="small"),
        "链接": st.column_config.LinkColumn("链接", display_text="🔗"),
        "原始文本": st.column_config.TextColumn(
            "原始参考文献",
            help="鼠标悬停可查看完整的参考文献文本。",
            width="medium",
        ),
        "状态": st.column_config.TextColumn(help="参考文献的校验状态。"),
        "说明": st.column_config.TextColumn(help="关于校验状态的说明。"),
        "替换建议": st.column_config.TextColumn(help="为无法验证的参考文献提供的替换建议。"),
    }

    df_display = df[['作者', '年份', '标题', '原始文本', '状态', '说明', '链接', '替换建议']]
    placeholder.dataframe(df_display, use_container_width=True, column_config=column_config)

    verified_count = 0
    warning_count = 0
    total_refs = len(references)

    for index, ref_object in enumerate(references):
        progress_text.text(f"正在验证 {index + 1}/{total_refs} | 已验证: {verified_count} | 未找到: {warning_count}")
        
        # 调用新的、统一的智能验证函数
        result: VerificationResult = verify_reference_with_search(ref_object)

        df.loc[index, "状态"] = status_emoji.get(result.status.value)
        df.loc[index, "说明"] = result.explanation
        df.loc[index, "链接"] = result.url

        if result.status == ReferenceStatus.VALIDATED:
            verified_count += 1
        else:
            warning_count += 1
            # 如果未找到，则获取替换建议
            suggestion = find_replacement_reference(ref_object)
            df.loc[index, "替换建议"] = suggestion
    
    # 所有处理完成后，更新最终的统计信息和表格
    progress_text.text(f"处理完成！ | 已验证: {verified_count} | 未找到: {warning_count}")
    df_display = df[['作者', '年份', '标题', '原始文本', '状态', '说明', '链接', '替换建议']]
    placeholder.dataframe(df_display, use_container_width=True, column_config=column_config)

    return df


def main():
    st.set_page_config(
        page_title="VeriExCite: 参考文献核验工具",
        page_icon="🔍",
        layout="wide",
    )

    st.title("VeriExCite: 参考文献核验工具")
    st.write(
        "本工具旨在帮助您核验学术论文（PDF格式）中引用的参考文献是否存在。它会自动提取文献列表，解析每个条目，并验证其有效性。"
    )

    with st.sidebar:
        st.header("输入")
        pdf_files = st.file_uploader("上传一个或多个 PDF 文件", type="pdf", accept_multiple_files=True)

        st.markdown(
            "您可以在 [Google AI Studio](https://ai.google.dev/) 免费申请 Gemini API 密钥，每天享有 1500 次请求的免费额度。"
        )
        api_key = st.text_input("请输入您的 Google Gemini API 密钥:", type="password")

    if st.sidebar.button("开始验证"):
        if not pdf_files:
            st.warning("请至少上传一个 PDF 文件。")
            return

        if not api_key:
            st.warning("请输入您的 Google Gemini API 密钥。")
            return

        try:
            set_google_api_key(api_key)
            all_results = []

            for pdf_file in pdf_files:
                st.subheader(f"正在处理: {pdf_file.name}")
                # 使用新的文本提取函数
                pdf_content = extract_text_from_uploaded_file(pdf_file)
                bib_text = extract_bibliography_section(pdf_content)

                with st.expander(f"从 {pdf_file.name} 提取的参考文献文本"):
                    st.text_area("提取内容", bib_text, height=200, label_visibility="hidden")

                results_df = process_and_verify(bib_text)
                if not results_df.empty:
                    results_df['来源文件'] = pdf_file.name
                    all_results.append(results_df)
                st.success(f"已完成: {pdf_file.name}")

            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                csv = combined_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="下载所有结果 (CSV)",
                    data=csv,
                    file_name='VeriExCite_分析结果.csv',
                    mime='text/csv',
                )

        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"处理过程中发生意外错误: {e}")


if __name__ == "__main__":
    main()