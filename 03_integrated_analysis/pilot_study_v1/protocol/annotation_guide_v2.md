# 标注规范 v2

本规范用于随后的真人试用。本轮所有科学答案保持空白。`workflow_status=pending` 只表示还没有人填写，不是科学判断。

先看 phase_a 的原文和中性问题，写下参考判断并保存。锁定之后才打开 phase_b 的历史模型输出。同一个人先 A 后 B，不能消除记忆影响。这些开发案例此前已经被讨论，隐藏当前模型答案不等于保证评阅者从未见过它们。试用标签不并入正式评价。

## 分开填写的字段

内容角色 `discourse_role_reference`：own_result、author_interpretation、background、unclear。这只回答这句话是本文结果、作者解释还是背景。

拆分 `segmentation_assessability`：能否判断拆分。`segmentation_flags` 可以同时包含 over_split 和 over_merged。已经看过并且没有问题时，标记为空列表，并保留 assessable。不要把“本文结果”和“拆分合理”放进同一个单选题。

实体身份 `identity`：这个对象是不是本次任务要看的基因或疾病相关对象。实体形式 `role_form`：RNA、蛋白、配体、受体、可溶性形式或干预构建体有没有写对。同一基因的 RNA 和蛋白不是身份错误。别名不同也不自动是另一个实体。CD137L 和 CD137-Fc 不作为 TNFRSF9 的等价名称来预填。

原文支持 `text_support`：在给出的完整标题和摘要里，全文是否支持这条 finding 或判断。允许 full_support、partial_support、contradiction、insufficient_text。模型没有给引用时，只要摘要还在，就不能因此把 text_support 填成信息不足。

引用支持 `citation_support`：模型实际给出的那条引用是否支持它声明的字段。没有引用时填 citation_missing。旧输出没有字段连接时填 field_link_unknown。定位到原文不等于语义支持。

比较范围 `comparison_scope`：report_pattern_only 或 aligned_comparator_and_outcome。检测到与未检测到、定位不同、相对参照升降、生存关联，是不同的报告模式。报告模式不同不等于同一参照下方向相反。

未评价不要用来表示文本不足。不适用不要用来表示技术失败。技术失败由程序记录，不由标注者改成 unassessable。

英文例子只出现在 phase_b，并且标明 non_gold。phase_a 不放模型句子。
