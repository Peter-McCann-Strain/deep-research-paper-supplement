# Case study 4: Maximum judge disagreement — base_p0

**Query ID:** `dsqa_0080`
**Source:** deepsearch_qa
**Domain:** Linguistics
**Difficulty:** complex

## Query text

> Which of Isabelle Darcy's (2012) coauthors on the article "Direct Mapping of Acoustics to Phonology: On the Lexical Encoding of Front Rounded Vowels in L1 English-- L2 French acquisition" has taught courses on both the German language and constructed languages at Indiana University-Bloomington?

## 3-judge mean overall score per pattern (base patterns only)

| Pattern | Mean overall score |
|:---|---:|
| base_p0 | 0.576 |
| base_p1 | 0.671 |
| base_p2 | 0.280 |
| base_p3 | 0.441 |
| base_p4 | 0.441 |
| base_p5 | 0.547 |
| base_p6 | 0.474 |
| base_p7 | 0.506 |
| base_p8 | 0.485 |
| base_p9 | 0.014 |
| base_p10 | 0.366 |

## Key-dimension verdicts

**base_p0**

| Judge | Dimension | Satisfied | Reasoning (truncated) |
|:---|:---|:---:|:---|
| claude_sonnet | analytical_depth | Y | Sources are synthesized into a unified investigation rather than summarized sequentially. |
| claude_sonnet | analytical_depth | Y | Analysis goes beyond surface-level identification to explain the reasoning behind the conclusion. |
| claude_sonnet | analytical_depth | Y | Meaningful connections between Sprouse's academic background and teaching activities are drawn. |
| claude_sonnet | analytical_depth | Y | Confirmed facts and uncertain inferences are clearly distinguished. |
| gpt52 | analytical_depth | N | The report largely summarizes and asserts rather than synthesizing corroborating details across multiple independent sources. |
| gpt52 | analytical_depth | N | It does not go beyond surface description to analyze evidence quality/trade-offs for the actual research question. |
| gpt52 | analytical_depth | N | Connections between relevant aspects (coauthor identity, IU teaching records, constructed language courses) are not substantively drawn with |
| gpt52 | analytical_depth | N | The report does not distinguish well-established facts from unverified assertions, especially for the key teaching claim. |
| claude_sonnet | citation_quality | Y | Claims are consistently attributed with inline citations. |
| claude_sonnet | citation_quality | Y | Citation format is consistent. |
| claude_sonnet | citation_quality | N | 3 sources does not meet the minimum requirement of 5 distinct sources. |
| claude_sonnet | citation_quality | Y | Multiple sources are used across different sections. |
| gpt52 | citation_quality | N | Key claims are not properly attributable to specific, named source passages with sufficient precision. |
| gpt52 | citation_quality | Y | Citation formatting is consistent within the report. |
| gpt52 | citation_quality | N | Fewer than the minimum 5 distinct sources are cited for the topic scope. |
| gpt52 | citation_quality | N | Different sections do not clearly draw on different sources, and the central claim is not independently corroborated. |
| claude_sonnet | factual_accuracy | Y | Factual claims about the article, coauthors, and methodology are accurate. |
| claude_sonnet | factual_accuracy | Y | Technical linguistics and psycholinguistics terminology is used precisely. |
| claude_sonnet | factual_accuracy | Y | Specific dates and numbers cited are accurate. |
| claude_sonnet | factual_accuracy | Y | Chronological ordering is correct. |
| claude_sonnet | factual_accuracy | Y | Comparative claims about coauthors' qualifications are supported by evidence from sources. |
| claude_sonnet | factual_accuracy | Y | No contradictions found. |
| claude_sonnet | factual_accuracy | Y | Limitations of the source coverage are accurately and honestly represented. |
| claude_sonnet | factual_accuracy | Y | Current state of the research area is appropriately contextualized. |
| gpt52 | factual_accuracy | N | The coauthor list and the claim about Sprouse teaching constructed languages at IU are not verified with authoritative sources and may be in |
| gpt52 | factual_accuracy | Y | Technical SLA/phonology terminology is used in a generally correct and precise way. |
| gpt52 | factual_accuracy | N | Specific dates are asserted but not supported with precise, checkable citations to the underlying documents. |
| gpt52 | factual_accuracy | Y | No complex historical timeline is presented, and the limited chronology described is internally plausible and consistent. |
| gpt52 | factual_accuracy | N | The comparative claim (“only coauthor”) is not supported by comprehensive evidence across all coauthors’ teaching records. |
| gpt52 | factual_accuracy | Y | There are no clear internal contradictions across sections. |
| gpt52 | factual_accuracy | N | Limitations discussed are about the phonology study, not the evidence/method used to answer the teaching-at-IU question, which is the releva |
| gpt52 | factual_accuracy | N | State-of-the-art is not identified where relevant to the query (which is primarily an attribution/biographical verification task). |
| claude_sonnet | information_recall | Y | All key facts needed to answer the research question are retrieved and included. |
| claude_sonnet | information_recall | Y | Relevant quantitative data is present, appropriate for this factual academic inquiry. |
| claude_sonnet | information_recall | Y | The most authoritative sources for identifying coauthors and their affiliations are included. |
| claude_sonnet | information_recall | Y | The systematic review of all coauthors ensures no critical evidence is overlooked. |
| gpt52 | information_recall | N | The report states an answer but does not actually retrieve/provide the key verifying fact (a quoted IU course listing or faculty profile) sh |
| gpt52 | information_recall | N | No quantitative data relevant to the query (teaching courses) is included. |
| gpt52 | information_recall | N | The most authoritative sources for “taught courses at IU” would be IU course catalogs, departmental teaching pages, or official faculty CVs, |
| gpt52 | information_recall | N | Widely available critical evidence (IU course catalog/department listings) is omitted, and the report itself acknowledges this gap. |

## Report excerpts (first ~200 words)

### base_p0

```text
# Identifying Isabelle Darcy's Coauthor Who Taught German and Constructed Languages at Indiana University-Bloomington ## Abstract This research report investigates which of Isabelle Darcy's coauthors on the 2012 article titled *"Direct Mapping of Acoustics to Phonology: On the Lexical Encoding of Front Rounded Vowels in L1 English–L2 French Acquisition"* has taught courses on both the German language and constructed languages at Indiana University-Bloomington. The article, coauthored by Laurent Dekydtspotter, Rex A. Sprouse, Justin Glover, Christiane Kaden, Michael McGuire, and John H. G. Scott, explores the acquisition of French front rounded vowels by native English speakers and introduces the "direct mapping from acoustics to phonology" (DMAP) approach. Based on the provided source evidence, this report identifies Rex A. Sprouse as the coauthor who has taught both German and constructed languages at Indiana University-Bloomington. The report provides a detailed analysis of the sources, methodology, and findings, as well as the limitations of the evidence used to address the research query. --- ## Introduction The purpose of this report is to determine which of Isabelle Darcy's coauthors on the 2012 article *"Direct Mapping of Acoustics to Phonology: On the Lexical Encoding of Front Rounded Vowels in L1 English–L2 French Acquisition"* has taught courses ...
```

## Interpretation

Judges disagree by std=0.381 on base_p0's output for this query. Inspection of per-judge verdicts shows different strictness thresholds on citation and factual-accuracy criteria: gpt52 tends to mark 'Web Search Synthesis' placeholders as failures, while sonnet accepts them when the surrounding prose is plausible. This motivates the decision to use multi-judge averaging with reported dispersion.
