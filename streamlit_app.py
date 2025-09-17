import streamlit as st
# 导入更新后的后端函数
from veriexcite import (
    extract_text_from_pdf,
    extract_bibliography_section,
    split_references,
    search_title,
    find_replacement_reference,
    set_google_api_key,
    ReferenceStatus,
    ReferenceCheckResult,
    ReferenceExtraction,
)
import io
import pandas as pd
import os
from typing import List

# 使用 veriexcite.py 中的 PDF 文本提取功能
def extract_text_from_uploaded_file(pdf_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """使用后端的 PyPDF2 模块从上传的文件中提取文本。"""
    if not pdf_file.name.lower().endswith(".pdf"):
        raise ValueError("上传的文件不是 PDF 格式。")

    pdf_bytes = pdf_file.getvalue()
    
    # 为了让 PyPDF2 能够处理，需要先将字节流写入一个临时文件
    import tempfile
    temp_dir = tempfile.gettempdir()
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    temp_pdf_path = os.path.join(temp_dir, pdf_file.name)
    
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)
        
    # 调用 veriexcite.py 中基于 PyPDF2 的函数
    return extract_text_from_pdf(temp_pdf_path)


def display_replacement_suggestions_for_file(results_df: pd.DataFrame, file_name: str):
    """Display replacement suggestions for a specific file in the main page"""
    # Filter for references that need replacement (both invalid and not found)
    warning_refs = results_df[results_df['状态'].isin(['未找到', '无效'])]
    
    if len(warning_refs) > 0:
        st.subheader(f"📋 {file_name} - 替换建议")
        
        # Show summary
        st.info(f"发现 {len(warning_refs)} 个需要替换的参考文献")
        
        for idx, (_, ref) in enumerate(warning_refs.iterrows()):
            with st.expander(f"🔍 建议 {idx + 1}: {ref['标题'][:60]}...", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**📄 原始文献:**")
                    st.write(ref['原始文本'])
                    st.write(f"**❌ 问题:** {ref['说明']}")
                
                with col2:
                    st.write(f"**👤 作者:** {ref['作者']}")
                    st.write(f"**📅 年份:** {ref['年份']}")
                    if ref['链接']:
                        st.write(f"**🔗 链接:** {ref['链接']}")
                
                st.markdown("---")
                
                if ref['替换建议'] and ref['替换建议'] != "未找到合适的替换建议":
                    st.write("**💡 替换建议:**")
                    
                    # Parse and display suggestions nicely
                    suggestions_text = ref['替换建议']
                    if "个替换建议" in suggestions_text:
                        # Split by suggestions
                        parts = suggestions_text.split("建议 ")
                        if len(parts) > 1:
                            # Extract reasoning
                            reasoning_part = parts[0]
                            reasoning = reasoning_part.split("推荐理由: ")[-1] if "推荐理由: " in reasoning_part else "基于主题分析搜索三个学术数据库"
                            st.write(f"**推荐理由:** {reasoning}")
                            
                            # Display each suggestion
                            for i, part in enumerate(parts[1:], 1):
                                if "匹配度:" in part and "文献:" in part:
                                    lines = part.strip().split('\n')
                                    if len(lines) >= 3:
                                        # Extract source and score from first line
                                        first_line = lines[0]
                                        if " - " in first_line and "匹配度:" in first_line:
                                            source = first_line.split(" - ")[1].split(" (匹配度:")[0]
                                            score = first_line.split("匹配度: ")[1].split("/100")[0]
                                            
                                            st.write(f"**建议 {i} - {source}:**")
                                            st.write(f"📄 {lines[1].replace('文献: ', '')}")
                                            st.write(f"🔗 {lines[2].replace('链接: ', '')}")
                                            st.write(f"⭐ 匹配度: {score}/100")
                                            st.markdown("---")
                    else:
                        st.write(suggestions_text)
                else:
                    st.write("**替换建议:** 暂无合适建议")
    else:
        st.success(f"✅ {file_name} - 所有参考文献验证通过，无需替换建议")

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
        "validated": "已验证",
        "invalid": "无效",
        "not_found": "未找到",
    }

    results = [
        {
            "作者": ref.author,
            "年份": str(ref.year),
            "标题": ref.title,
            "原始文本": ref.bib,
            "状态": "验证中...",
            "说明": "待处理",
            "链接": "",
            "替换建议": "",
        }
        for ref in references
    ]
    df = pd.DataFrame(results)

    column_config = {
        "作者": st.column_config.TextColumn("第一作者", help="第一作者的姓氏或机构名称。"),
        "年份": st.column_config.TextColumn(" 年份", width="small"),
        "链接": st.column_config.LinkColumn("链接", display_text="查看"),
        "原始文本": st.column_config.TextColumn(
            "📄 原始参考文献",
            help="鼠标悬停可查看完整的参考文献文本。",
            width="medium",
        ),
        "状态": st.column_config.TextColumn("状态", help="参考文献的校验状态。"),
        "说明": st.column_config.TextColumn("说明", help="关于校验状态的说明。"),
        "替换建议": st.column_config.TextColumn("替换建议", help="为无法验证的参考文献提供的替换建议。"),
    }

    df_display = df[['作者', '年份', '标题', '原始文本', '状态', '说明', '链接', '替换建议']]
    placeholder.dataframe(df_display, use_container_width=True, column_config=column_config)

    verified_count = 0
    warning_count = 0
    total_refs = len(references)

    for index, ref_object in enumerate(references):
        progress_text.text(f"正在验证 {index + 1}/{total_refs} | 已验证: {verified_count} | 未找到: {warning_count}")
        
        try:
            result: ReferenceCheckResult = search_title(ref_object)
            if result is None:
                result = ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Search returned no result.")
        except Exception as e:
            st.error(f"Error processing reference: {e}")
            result = ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Processing error: {e}")

        df.loc[index, "状态"] = status_emoji.get(result.status.value)
        df.loc[index, "说明"] = result.explanation
        df.loc[index, "链接"] = ref_object.URL if hasattr(ref_object, 'URL') else ""

        if result.status == ReferenceStatus.VALIDATED:
            verified_count += 1
        else:
            warning_count += 1
            # Show progress for replacement suggestions
            progress_container = st.empty()
            progress_text = progress_container.text(f"正在为 '{ref_object.title}' 寻找替换建议...")
            
            def update_progress(message):
                progress_text.text(message)
            
            suggestion = find_replacement_reference(ref_object, progress_callback=update_progress)
            
            # Clear progress text
            progress_container.empty()
            
            if suggestion.found:
                # Format three suggestions nicely for CSV export
                suggestion_count = sum([1 for s in [suggestion.suggestion1_bib, suggestion.suggestion2_bib, suggestion.suggestion3_bib] if s.strip()])
                suggestion_text = f"找到 {suggestion_count} 个替换建议\n"
                suggestion_text += f"推荐理由: {suggestion.reasoning}\n\n"
                
                # Add arXiv suggestion
                if suggestion.suggestion1_bib and suggestion.suggestion1_bib.strip():
                    suggestion_text += f"建议 1 - {suggestion.suggestion1_source} (匹配度: {suggestion.suggestion1_score}/100):\n"
                    suggestion_text += f"文献: {suggestion.suggestion1_bib}\n"
                    suggestion_text += f"链接: {suggestion.suggestion1_url}\n\n"
                
                # Add Crossref suggestion
                if suggestion.suggestion2_bib and suggestion.suggestion2_bib.strip():
                    suggestion_text += f"建议 2 - {suggestion.suggestion2_source} (匹配度: {suggestion.suggestion2_score}/100):\n"
                    suggestion_text += f"文献: {suggestion.suggestion2_bib}\n"
                    suggestion_text += f"链接: {suggestion.suggestion2_url}\n\n"
                
                # Add Google Scholar suggestion
                if suggestion.suggestion3_bib and suggestion.suggestion3_bib.strip():
                    suggestion_text += f"建议 3 - {suggestion.suggestion3_source} (匹配度: {suggestion.suggestion3_score}/100):\n"
                    suggestion_text += f"文献: {suggestion.suggestion3_bib}\n"
                    suggestion_text += f"链接: {suggestion.suggestion3_url}\n\n"
                
                df.loc[index, "替换建议"] = suggestion_text
            else:
                df.loc[index, "替换建议"] = f"未找到合适的替换建议\n原因: {suggestion.reasoning}"
    
    progress_text.text(f"处理完成！ | 已验证: {verified_count} | 需要替换: {warning_count}")
    
    # Clear the progress text and placeholder after completion
    progress_text.empty()
    placeholder.empty()

    return df


def main():
    st.set_page_config(
        page_title="学佑星途: 参考文献核验工具",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header with better styling
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">论文检查工具</h1>
        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;">智能参考文献核验与替换建议工具</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <h3 style="margin-top: 0; color: #1976d2;">功能特点</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li><strong>智能解析</strong>：自动提取PDF中的参考文献列表</li>
                <li><strong>多源验证</strong>：通过Crossref、arXiv、Google Scholar等验证文献真实性</li>
                <li><strong>智能替换</strong>：AI分析主题后从三个数据库提供替换建议</li>
                <li><strong>多语言支持</strong>：支持中、英、日、法、德、西、俄、意、葡、韩等8+语言</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        
        pdf_files = st.file_uploader(
            "上传PDF文件", 
            type="pdf", 
            accept_multiple_files=True,
            help="支持上传多个PDF文件进行批量处理"
        )

        st.markdown("---")
        
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h4 style="margin-top: 0; color: #f57c00;">API密钥</h4>
            <p style="margin-bottom: 0; font-size: 0.9rem;">需要Google Gemini API密钥来生成替换建议</p>
        </div>
        """, unsafe_allow_html=True)
        
        api_key = st.text_input(
            "Google Gemini API 密钥:", 
            type="password",
            help="在 [Google AI Studio](https://ai.google.dev/) 免费申请"
        )
        
        if api_key:
            st.success("API密钥已设置")
        else:
            st.warning("请设置API密钥以获取替换建议")

        st.markdown("---")
        
        # Status indicator
        if st.session_state.get('verification_completed', False):
            st.success("✅ 验证已完成")
        elif st.session_state.get('start_verification', False):
            st.success("✅ 验证进行中...")
        else:
            st.info("⏳ 等待开始验证")
        
        st.markdown("---")
        
        # Start verification button
        if st.button("🚀 开始验证", type="primary", use_container_width=True):
            st.session_state.start_verification = True
            st.rerun()
        
        # Reset button
        if st.button("🔄 重新开始", use_container_width=True):
            st.session_state.start_verification = False
            st.session_state.verification_completed = False
            if 'all_results' in st.session_state:
                del st.session_state.all_results
            st.rerun()

    # Main processing area
    if pdf_files and api_key:
        if not st.session_state.get('start_verification', False):
            st.info("👆 请点击侧边栏的 '🚀 开始验证' 按钮开始处理")
        elif st.session_state.get('start_verification', False) and not st.session_state.get('verification_completed', False):
            try:
                set_google_api_key(api_key)
                all_results = []

                for pdf_file in pdf_files:
                    st.subheader(f"正在处理: {pdf_file.name}")
                    pdf_content = extract_text_from_uploaded_file(pdf_file)
                    bib_text = extract_bibliography_section(pdf_content)

                    with st.expander(f"从 {pdf_file.name} 提取的参考文献文本"):
                        st.text_area("提取内容", bib_text, height=200, label_visibility="hidden")

                    results_df = process_and_verify(bib_text)
                    if not results_df.empty:
                        results_df['来源文件'] = pdf_file.name
                        all_results.append(results_df)
                        
                        # Display replacement suggestions for this file
                        display_replacement_suggestions_for_file(results_df, pdf_file.name)
                        
                    st.success(f"已完成: {pdf_file.name}")
                    st.markdown("---")

                if all_results:
                    # Store results in session state for persistence
                    st.session_state.all_results = all_results
                    st.session_state.verification_completed = True
                    
                    # Show completion message
                    st.success("🎉 所有文件验证完成！")

            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"处理过程中发生意外错误: {e}")
        
        # Show results if verification is completed
        if st.session_state.get('verification_completed', False) and 'all_results' in st.session_state:
            all_results = st.session_state.all_results
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # Display all three replacement suggestions in the CSV
            st.subheader("分析结果")
            
            # Show summary statistics
            total_refs = len(combined_results)
            verified_refs = len(combined_results[combined_results['状态'] == '已验证'])
            invalid_refs = len(combined_results[combined_results['状态'] == '无效'])
            not_found_refs = len(combined_results[combined_results['状态'] == '未找到'])
            warning_refs = invalid_refs + not_found_refs
            success_rate = (verified_refs / total_refs * 100) if total_refs > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总参考文献", total_refs)
            with col2:
                st.metric("已验证", verified_refs)
            with col3:
                st.metric("需要替换", warning_refs)
            with col4:
                st.metric("成功率", f"{success_rate:.1f}%")
            
            st.markdown("---")
            
            # Show the results table with better formatting
            st.dataframe(
                combined_results, 
                use_container_width=True,
                column_config={
                    "作者": st.column_config.TextColumn("第一作者"),
                    "年份": st.column_config.TextColumn("年份", width="small"),
                    "标题": st.column_config.TextColumn("标题", width="medium"),
                    "原始文本": st.column_config.TextColumn("原始参考文献", width="large"),
                    "状态": st.column_config.TextColumn("状态"),
                    "说明": st.column_config.TextColumn("说明", width="medium"),
                    "链接": st.column_config.LinkColumn("链接", display_text="查看"),
                    "替换建议": st.column_config.TextColumn("替换建议", width="large"),
                    "来源文件": st.column_config.TextColumn("来源文件")
                }
            )
            
            # Download button
            csv = combined_results.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label="下载所有结果 (CSV)",
                data=csv,
                file_name='分析结果.csv',
                mime='text/csv',
                type="primary"
            )
    elif pdf_files and not api_key:
        st.warning("请设置API密钥以开始处理")
    elif not pdf_files:
        st.info("请上传PDF文件并设置API密钥以开始处理")


if __name__ == "__main__":
    main()