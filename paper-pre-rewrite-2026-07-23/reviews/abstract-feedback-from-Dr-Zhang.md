# Abstract feedback from Dr. Zhang

Hi Robert,

Thank you for sending the draft. **Below are my comments:**

For the abstract, consider avoiding terms such as “demand model,” since the paper focuses on an imitation learning model rather than demand modeling. A clear abstract typically follows this structure: first introduce the problem, then explain the motivation and research gap, present the proposed solution, and finally summarize the experimental results. For example, the abstract could conclude by stating that experiments on several datasets demonstrate the effectiveness of the proposed approach.

For the main visualization, consider reducing the figure to approximately half-page width and retaining only Part (c). The current square and cross symbols may not communicate the intended message intuitively. More meaningful icons, such as passengers and taxi drivers, could better illustrate the resource redistribution process and make the figure easier to understand.

In the introduction, ensure that all related approaches and claims are properly cited. For example, the following paragraph discusses several categories of methods and therefore requires appropriate references:

“Interventions typically target one end of the pipeline or the other. In-processing methods regularize the model while leaving the demonstrations biased [34], so the learning objective and the training signal pull in different directions throughout training. Data-generation or data-rebalancing methods modify the demonstration distribution, but replacing or extensively altering the original data may reduce realism, introduce distribution shifts, and make it difficult to identify the source of the fairness improvement. FAMAIL takes a third approach.”

The terminology should also be aligned with language commonly used by the machine learning community. Terms such as “generative repair” may not be immediately understandable or widely recognized. More precise terms, such as synthetic data generation, data augmentation, demonstration rebalancing, or bias mitigation through data modification, should be used depending on the specific methods being discussed. Please rely primarily on the terminology and categorization established through the literature review rather than adopting unfamiliar terms suggested by an LLM. Using established terminology will help readers understand how the proposed method relates to existing research and clarify the paper’s contribution.

To improve the organization of the introduction, consider following this sequence:

1. Introduce the general application scenario and explain its importance.
2. Present the main motivation: imitation learning models may inherit and amplify biases contained in expert demonstrations.
3. Categorize existing approaches and explain their limitations.
4. Use a visualization or motivating example to highlight the unresolved research gap.
5. Introduce the proposed approach and explain how it addresses this gap.
6. Conclude the introduction with a concise list of the paper’s main contributions.

For the discussion of existing approaches, it may be helpful to present their limitations explicitly and systematically. Rather than only describing what each category of methods does, explain why it cannot fully address the problem considered in this paper and how the proposed approach differs. This will make the research gap and technical novelty more convincing.

The related-work section should follow a similar structure. After discussing each group of related methods, include a brief summary of its limitations in the context of this paper and clearly distinguish those methods from the proposed approach. This will strengthen the connection between the literature review and the paper’s contributions.

Finally, consider using brighter and more visually engaging colors in the figures. A more cheerful and consistent color palette may improve readability and make the visualizations more appealing.

**You may initiate a paper submission with the title** "Mitigating Demonstration Bias via Fairness-Aware Trajectory Editing". The author list is You, Manuel, Charles, Dr. Xin Wang, Dr. Yanhua Li ([yli15@wpi.edu](mailto:yli15@wpi.edu)), Dr. Kash, me.

Many Thanks,

Xin Zhang

Assistant Professor

Computer Science Department

San Diego State University