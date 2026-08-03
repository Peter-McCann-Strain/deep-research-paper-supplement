# Case study 2: Top-cluster tie — P1 (Iterative RAG) edges P4

**Query ID:** `dsqa_0063`
**Source:** deepsearch_qa
**Domain:** Arts
**Difficulty:** complex

## Query text

> What was the first year in which any of the nine artists that formed "Group X" that exhibited at the Mansard Gallery received a Blue Plaque in London? For this exercise, consider the nine members to be Lewis, Dismorr, Etchells, Hamilton, Roberts, Wadsworth, Dobson, Ginner and Kauffer.

## 3-judge mean overall score per pattern (base patterns only)

| Pattern | Mean overall score |
|:---|---:|
| base_p0 | 0.518 |
| base_p1 | 0.761 |
| base_p2 | 0.600 |
| base_p3 | 0.643 |
| base_p4 | 0.396 |
| base_p5 | 0.626 |
| base_p6 | 0.731 |
| base_p7 | 0.706 |
| base_p8 | 0.553 |
| base_p9 | 0.014 |
| base_p10 | 0.173 |

## Key-dimension verdicts

**base_p1**

| Judge | Dimension | Satisfied | Reasoning (truncated) |
|:---|:---|:---:|:---|
| claude_sonnet | analytical_depth | Y | Synthesis across sources is performed. |
| claude_sonnet | analytical_depth | N | Analysis remains at the descriptive/factual level. |
| claude_sonnet | analytical_depth | N | No analytical connections drawn. |
| claude_sonnet | analytical_depth | Y | Established vs uncertain claims are distinguished. |
| gpt52 | analytical_depth | Y | It integrates information across sources to produce a single earliest-year determination rather than only listing sources separately. |
| gpt52 | analytical_depth | N | The report is mostly descriptive and does not deeply analyze mechanisms or trade-offs relevant to the outcome. |
| gpt52 | analytical_depth | N | Connections between different aspects (e.g., eligibility rules vs. why certain members lack plaques) are not substantively developed. |
| gpt52 | analytical_depth | Y | It signals which claims are established (specific plaques/years) and which are uncertain (absence of plaques for others). |
| claude_sonnet | citation_quality | Y | Claims are cited. |
| claude_sonnet | citation_quality | Y | Consistent format. |
| claude_sonnet | citation_quality | Y | Sufficient sources. |
| claude_sonnet | citation_quality | N | Sections tend to rely on single sources per claim. |
| gpt52 | citation_quality | Y | Key claims are attributed with inline bracketed citations to named references. |
| gpt52 | citation_quality | Y | Citation formatting is consistent throughout the report. |
| gpt52 | citation_quality | Y | The number of sources is more than sufficient for the scope (minimum 5). |
| gpt52 | citation_quality | Y | The report draws on multiple sources across sections rather than relying on a single reference. |
| claude_sonnet | factual_accuracy | Y | Core factual claims are accurate. |
| claude_sonnet | factual_accuracy | Y | Terminology is accurate. |
| claude_sonnet | factual_accuracy | Y | Specific dates and numbers appear accurate. |
| claude_sonnet | factual_accuracy | Y | Chronological ordering is accurate. |
| claude_sonnet | factual_accuracy | Y | Comparative chronological claims are supported. |
| claude_sonnet | factual_accuracy | Y | No contradictions identified. |
| claude_sonnet | factual_accuracy | Y | Limitations are accurately represented. |
| claude_sonnet | factual_accuracy | N | Some uncertainty remains about whether the 'no plaque' status for remaining members is fully verified via current official database. |
| gpt52 | factual_accuracy | N | Several factual assertions are made without sufficient corroboration in the report itself (and some are potentially dubious), so overall acc |
| gpt52 | factual_accuracy | Y | Key terms (Blue Plaque scheme, installation bodies) are used in a generally correct and precise way. |
| gpt52 | factual_accuracy | N | Not all specific dates are supported by equally reliable citations, so date accuracy is not clearly demonstrated for every numeric claim. |
| gpt52 | factual_accuracy | Y | The internal timeline of plaque years is coherent and ordered correctly relative to the report’s own evidence. |
| gpt52 | factual_accuracy | Y | The comparative claim that Lewis predates the others is supported by cited installation years. |
| gpt52 | factual_accuracy | Y | There are no conflicting statements about the main result across sections. |
| gpt52 | factual_accuracy | N | The limitations are not fully accurate as stated and do not precisely reflect the available resources the report itself uses. |
| gpt52 | factual_accuracy | N | The query does not require state-of-the-art discussion, and the report does not clearly identify any relevant “state-of-the-art” beyond gene |
| claude_sonnet | information_recall | Y | Key facts to answer the query are present and appear correct. |
| claude_sonnet | information_recall | Y | Relevant quantitative data are present. |
| claude_sonnet | information_recall | Y | Official primary sources are cited. |
| claude_sonnet | information_recall | N | The verification that other 5 members have no Blue Plaques relies on absence of evidence rather than a comprehensive database check. |
| gpt52 | information_recall | Y | It states the earliest year and identifies which of the nine artists received it, directly answering the query. |
| gpt52 | information_recall | Y | The report includes the key quantitative data (years) needed for the question and additional relevant counts. |
| gpt52 | information_recall | Y | It relies on the most authoritative source for London Blue Plaques (English Heritage) for the central claims. |
| gpt52 | information_recall | N | The report does not demonstrate comprehensive checking (e.g., explicit searches in official registers for each name), so widely-available ev |

**base_p4**

| Judge | Dimension | Satisfied | Reasoning (truncated) |
|:---|:---|:---:|:---|
| claude_sonnet | analytical_depth | Y | The report synthesizes findings across art history, cultural policy, and archival research into a coherent argument about delayed recognitio |
| claude_sonnet | analytical_depth | Y | The report identifies the mechanism of institutional conservatism as a driver of delayed recognition, going beyond surface description. |
| claude_sonnet | analytical_depth | Y | The report connects individual recognition to broader art movement validation and evolving historical narratives. |
| claude_sonnet | analytical_depth | Y | The report uses qualifiers such as 'Strong evidence suggests' to mark well-established claims versus interpretive assertions. |
| gpt52 | analytical_depth | N | Without multiple substantive sources and specific data, the report cannot credibly synthesize across evidence. |
| gpt52 | analytical_depth | N | The analysis remains largely conceptual and does not identify evidence-backed mechanisms or trade-offs relevant to the query. |
| gpt52 | analytical_depth | Y | The report draws connections between art-historical re-evaluation and public commemoration practices. |
| gpt52 | analytical_depth | N | It does not clearly separate well-established facts from speculative or contested claims with supporting citations. |
| claude_sonnet | citation_quality | Y | Factual claims carry inline citations, though most references are to only two sources. |
| claude_sonnet | citation_quality | Y | Citation format is consistent using bracketed numbers throughout. |
| claude_sonnet | citation_quality | N | Only 2 distinct sources are cited, well below the minimum of 5 required for appropriate coverage of this topic. |
| claude_sonnet | citation_quality | N | All sections uniformly cite the same two sources [1][2], demonstrating no diversity of source usage across sections. |
| gpt52 | citation_quality | N | Claims are not properly attributed to named, verifiable sources appropriate to the assertions being made. |
| gpt52 | citation_quality | N | Citation formatting is inconsistent and incomplete, preventing verification. |
| gpt52 | citation_quality | N | The topic scope requires at least five distinct sources, especially including English Heritage plaque records. |
| gpt52 | citation_quality | N | The report relies on the same minimal set of sources across sections rather than drawing on varied references. |
| claude_sonnet | factual_accuracy | Y | The membership list and exhibition details for Group X are accurately stated and consistent with historical records. |
| claude_sonnet | factual_accuracy | Y | Art historical and cultural policy terminology is used correctly throughout the report. |
| claude_sonnet | factual_accuracy | Y | The specific dates provided — 1866 for the Blue Plaque scheme's establishment and the 1920 exhibition dates — are historically accurate. |
| claude_sonnet | factual_accuracy | Y | The historical sequence — Vorticism pre-WWI, Group X 1920, late 20th century revival — is chronologically correct. |
| claude_sonnet | factual_accuracy | N | No comparative claims between Group X artists or between recognition timelines are made in the report, so this criterion cannot be satisfied |
| claude_sonnet | factual_accuracy | Y | All sections are internally consistent — no contradictions exist between the report's claims. |
| claude_sonnet | factual_accuracy | Y | The report accurately and thoroughly describes the archival and source limitations that prevent a definitive answer. |
| claude_sonnet | factual_accuracy | Y | The current management of the Blue Plaque scheme by English Heritage is correctly identified. |
| gpt52 | factual_accuracy | N | The report makes broad factual assertions without support and does not verify the central factual question, undermining accuracy. |
| gpt52 | factual_accuracy | Y | Terminology is generally used in a correct and conventional way for this topic. |
| gpt52 | factual_accuracy | N | The report fails to cite the specific date/benchmark required (the first Blue Plaque year). |
| gpt52 | factual_accuracy | N | The report does not establish any chronology for Blue Plaque awards and thus does not correctly answer the timeline asked in the query. |
| gpt52 | factual_accuracy | N | Comparative/timing claims are asserted without any cited evidence of actual plaque dates for any member. |
| gpt52 | factual_accuracy | Y | There are no clear internal contradictions across sections regarding the main conclusion of indeterminacy. |
| gpt52 | factual_accuracy | Y | The report explicitly states limitations and frames its conclusion as constrained by evidence gaps. |
| gpt52 | factual_accuracy | N | Where relevant, the report does not identify the current best/standard source of truth for Blue Plaque information. |
| claude_sonnet | information_recall | N | The core fact needed — the specific year of the first Blue Plaque — is not retrieved. The report explicitly concedes it cannot answer the qu |
| claude_sonnet | information_recall | N | No quantitative data about Blue Plaque award years, dates of recognition, or comparable metrics are present in the report. |
| claude_sonnet | information_recall | N | The only named source is a general encyclopedic explainer website, not primary sources such as English Heritage's official Blue Plaque regis |
| claude_sonnet | information_recall | N | English Heritage maintains a searchable online Blue Plaque database that would contain award years. This critical publicly-available resourc |
| gpt52 | information_recall | N | The report does not provide the key fact required by the query: the first year any of the nine artists received a London Blue Plaque. |
| gpt52 | information_recall | N | The only quantitative/date data relevant to the query would be the Blue Plaque year(s), which are missing. |
| gpt52 | information_recall | N | Authoritative sources for Blue Plaques (e.g., English Heritage plaque entries) are not used or cited. |
| gpt52 | information_recall | N | Widely available primary/authoritative evidence (English Heritage plaque records) is omitted, preventing a definitive answer. |

## Report excerpts (first ~200 words)

### base_p1

```text
# Research Report: The First Year a Member of "Group X" Received a Blue Plaque in London ## Abstract This research report investigates the first year in which any of the nine artists associated with "Group X"—a short-lived British artistic movement that exhibited at the Mansard Gallery in 1920—received an English Heritage Blue Plaque in London. The nine artists under consideration are Percy Wyndham Lewis, Jessica Dismorr, Frederick Etchells, Cuthbert Hamilton, William Roberts, Edward Wadsworth, Frank Dobson, Charles Ginner, and Edward McKnight Kauffer. The findings are based on a comprehensive analysis of 21 sources, including official reports, documentation, and secondary literature. The research concludes that Percy Wyndham Lewis was the first member of Group X to receive a Blue Plaque in London, with the plaque installed in 1983 at 61 Palace Gardens Terrace, Kensington, London [2]. The report outlines the methodology, key findings, and limitations of the research, while also discussing areas of ambiguity and disagreement in the available literature. --- ## Introduction The English Heritage Blue Plaque scheme, established in 1866, commemorates notable individuals by marking their former residences in London. The program recognizes individuals who have made significant contributions to society, provided they have been deceased for at ...
```

### base_p4

```text
# Tracing the Legacy of Group X: The First Blue Plaque Recognition and Its Cultural Significance ## Abstract The Blue Plaque scheme, a hallmark of cultural recognition in the United Kingdom, commemorates individuals who have made significant contributions to society. This report investigates the first year in which any of the nine artists from Group X—an influential collective of British modernists that exhibited at the Mansard Gallery in 1920—received a Blue Plaque in London. Drawing from multiple expert perspectives, this report examines the historical and cultural significance of Blue Plaques, the challenges in identifying the first recognition year for Group X artists, and the broader implications of their posthumous recognition. The findings reveal a complex interplay between the evolving appreciation of avant-garde art, the posthumous nature of Blue Plaque awards, and the limitations of archival documentation. This report concludes by emphasizing the importance of continued research to fill evidence gaps and further illuminate the legacy of Group X within the broader narrative of British modernism. --- ## Introduction: The Legacy of Group X and the Blue Plaque Scheme Group X, a short-lived but influential collective of nine artists, emerged in 1920 as a continuation of the pre-World War I Vorticist ...
```

## Interpretation

P1 (Iterative RAG) scores 0.761 vs P4 0.396 here (Δ=0.365). Iterative retrieval grounds claims with fresher/more-specific evidence whereas STORM's perspective expansion can dilute focus on factual queries. This example supports treating P1 and P4 as a top cluster rather than a strict P4 > P1 ordering.
