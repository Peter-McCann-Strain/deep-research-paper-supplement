# Analysis Dataframes — Coverage Report

- Total queries in manifest: **90**
- Patterns discovered: **73** (base + ablation + protocol_a + variance + disentanglement)
- Judges: claude_code, claude_opus, claude_sonnet, gpt52

`claude_code` is an archived frozen-data label retained for provenance. The
supported public judge workflow uses direct provider APIs and does not require
Claude Code or local assistant sessions.

## Executive summary

- The headline base-pattern reference is covered by the shipped compact tables and canonical store; missing historical cells are already reflected in the published `n_queries` counts.
- The public rebuild uses `data/analysis/*.parquet` and `paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json`; it does not need raw generated reports or raw judge packet trees.
- `ablation_p5_no_citation_verify` is excluded from statistical comparisons because the run stopped early and has only two judged reports.
- Missing-file sections below document historical raw artifacts that were not shipped. They explain provenance and coverage limits; they are not required for the supported public table/figure rebuild.

## Post-fix anomaly summary

- Rubric-drift groups (same (pattern,query,dim) with different criterion sets across judges): **349**
- Stored-vs-recomputed overall_score mismatches (|Δ|>0.01): **1045**
- Total raw anomalies logged: **2**

## Judge coverage per pattern (X / 90 queries scored)

| pattern                        |   claude_code |   claude_opus |   claude_sonnet |   gpt52 |
|:-------------------------------|--------------:|--------------:|----------------:|--------:|
| ablation_p3_no_quality_eval    |             0 |             3 |              87 |      87 |
| ablation_p3_no_topic_mining    |             0 |             0 |              86 |      86 |
| ablation_p4_fixed_perspectives |             0 |             0 |              90 |      90 |
| ablation_p4_no_conversations   |             0 |             8 |              90 |      90 |
| ablation_p4_no_triangulation   |             0 |             0 |              90 |      89 |
| ablation_p5_fixed_width        |             0 |             0 |              82 |      82 |
| ablation_p5_no_citation_verify |             0 |             0 |               2 |       0 |
| ablation_p5_no_meta_eval       |             0 |             0 |              86 |      86 |
| base_p0                        |             0 |            90 |              90 |      90 |
| base_p0_v1                     |             0 |             0 |               0 |      30 |
| base_p0_v10                    |             0 |             0 |               0 |      30 |
| base_p0_v11                    |             0 |             0 |               0 |      30 |
| base_p0_v2                     |             0 |             0 |               0 |      30 |
| base_p0_v3                     |             0 |             0 |               0 |      30 |
| base_p0_v4                     |             0 |             0 |               0 |      30 |
| base_p0_v5                     |             0 |             0 |               0 |      30 |
| base_p0_v6                     |             0 |             0 |               0 |      30 |
| base_p0_v7                     |             0 |             0 |               0 |      30 |
| base_p0_v8                     |             0 |             0 |               0 |      30 |
| base_p0_v9                     |             0 |             0 |               0 |      30 |
| base_p1                        |             0 |            89 |              89 |      90 |
| base_p10                       |             0 |            90 |              90 |      90 |
| base_p10_v1                    |             0 |             0 |               0 |      30 |
| base_p10_v2                    |             0 |             0 |               0 |      30 |
| base_p10_v3                    |             0 |             0 |               0 |      24 |
| base_p11                       |            79 |             0 |              80 |      89 |
| base_p11_16turn                |             0 |             0 |              27 |      29 |
| base_p12                       |            51 |             0 |              52 |      90 |
| base_p1_7b                     |             0 |             0 |               0 |      12 |
| base_p1_v1                     |             0 |             0 |               0 |      30 |
| base_p1_v2                     |             0 |             0 |               0 |      30 |
| base_p1_v3                     |             0 |             0 |               0 |      30 |
| base_p2                        |             0 |            90 |              90 |      90 |
| base_p3                        |             0 |            88 |              88 |      89 |
| base_p4                        |             0 |            90 |              90 |      90 |
| base_p4_7b                     |             0 |             0 |               0 |      12 |
| base_p4_v1                     |             0 |             0 |               0 |      30 |
| base_p4_v2                     |             0 |             0 |               0 |      30 |
| base_p4_v3                     |             0 |             0 |               0 |      29 |
| base_p5                        |             0 |            89 |              89 |      89 |
| base_p5_v1                     |             0 |             0 |               0 |      14 |
| base_p5_v2                     |             0 |             0 |               0 |       3 |
| base_p5_v3                     |             0 |             0 |               0 |       3 |
| base_p6                        |             0 |            87 |              87 |      87 |
| base_p6_v1                     |             0 |             0 |               0 |       8 |
| base_p6_v2                     |             0 |             0 |               0 |       3 |
| base_p6_v3                     |             0 |             0 |               0 |       3 |
| base_p7                        |             0 |            90 |              90 |      90 |
| base_p7_v1                     |             0 |             0 |               0 |      30 |
| base_p7_v2                     |             0 |             0 |               0 |      30 |
| base_p7_v3                     |             0 |             0 |               0 |      30 |
| base_p8                        |             0 |            90 |              90 |      90 |
| base_p8_v1                     |             0 |             0 |               0 |      18 |
| base_p8_v2                     |             0 |             0 |               0 |       6 |
| base_p8_v3                     |             0 |             0 |               0 |       6 |
| base_p9                        |             0 |            90 |              90 |      90 |
| disentangle_matched_p1         |             0 |             0 |               0 |      29 |
| disentangle_matched_p4         |             0 |             0 |               0 |       9 |
| oracle_t1_p0                   |             0 |            29 |              29 |      30 |
| oracle_t1_p1                   |             0 |            29 |              29 |      30 |
| oracle_t1_p2                   |             0 |             0 |               0 |      30 |
| oracle_t1_p3                   |             0 |             0 |               0 |      30 |
| oracle_t1_p4                   |             0 |            29 |              29 |      30 |
| oracle_t1_p5                   |             0 |            29 |              29 |      30 |
| oracle_t1_p6                   |             0 |            29 |              29 |      30 |
| oracle_t1_p7                   |             0 |            29 |              29 |      30 |
| oracle_t1_p8                   |             0 |            29 |              29 |      30 |
| protocol_a_tavily_p0           |            13 |             0 |               4 |      29 |
| protocol_a_tavily_p1           |            28 |             0 |               4 |      29 |
| protocol_a_tavily_p3           |            22 |             0 |               4 |      28 |
| protocol_a_tavily_p4           |            29 |             0 |               4 |      29 |
| protocol_a_tavily_p5           |            28 |             0 |               4 |      28 |
| protocol_a_tavily_p8           |            26 |             0 |               4 |      29 |


## Run / report existence per pattern

| pattern                        |   reports_exist |   checkpoints_found |   success | excluded   |
|:-------------------------------|----------------:|--------------------:|----------:|:-----------|
| ablation_p3_no_quality_eval    |              89 |                  90 |        89 | False      |
| ablation_p3_no_topic_mining    |              89 |                  90 |        89 | False      |
| ablation_p4_fixed_perspectives |              90 |                  90 |        90 | False      |
| ablation_p4_no_conversations   |              90 |                  90 |        90 | False      |
| ablation_p4_no_triangulation   |              90 |                  90 |        90 | False      |
| ablation_p5_fixed_width        |              90 |                  90 |        90 | False      |
| ablation_p5_no_citation_verify |              54 |                  55 |        54 | True       |
| ablation_p5_no_meta_eval       |              90 |                  90 |        90 | False      |
| base_p0                        |              90 |                  90 |        90 | False      |
| base_p0_v1                     |              30 |                  30 |        30 | False      |
| base_p0_v10                    |              30 |                  30 |        30 | False      |
| base_p0_v11                    |              30 |                  30 |        30 | False      |
| base_p0_v2                     |              30 |                  30 |        30 | False      |
| base_p0_v3                     |              30 |                  30 |        30 | False      |
| base_p0_v4                     |              30 |                  30 |        30 | False      |
| base_p0_v5                     |              30 |                  30 |        30 | False      |
| base_p0_v6                     |              30 |                  30 |        30 | False      |
| base_p0_v7                     |              30 |                  30 |        30 | False      |
| base_p0_v8                     |              30 |                  30 |        30 | False      |
| base_p0_v9                     |              30 |                  30 |        30 | False      |
| base_p1                        |              90 |                  90 |        90 | False      |
| base_p10                       |              90 |                  90 |        90 | False      |
| base_p10_v1                    |              30 |                  30 |        30 | False      |
| base_p10_v2                    |              30 |                  30 |        30 | False      |
| base_p10_v3                    |              30 |                  30 |        30 | False      |
| base_p11                       |              89 |                  90 |        89 | False      |
| base_p11_16turn                |              29 |                  30 |        29 | False      |
| base_p12                       |              90 |                  90 |        90 | False      |
| base_p1_7b                     |              12 |                  12 |        12 | False      |
| base_p1_v1                     |              30 |                  30 |        30 | False      |
| base_p1_v2                     |              30 |                  30 |        30 | False      |
| base_p1_v3                     |              30 |                  30 |        30 | False      |
| base_p2                        |              90 |                  90 |        90 | False      |
| base_p3                        |              89 |                  90 |        89 | False      |
| base_p4                        |              90 |                  90 |        90 | False      |
| base_p4_7b                     |              12 |                  12 |        12 | False      |
| base_p4_v1                     |              30 |                  30 |        30 | False      |
| base_p4_v2                     |              30 |                  30 |        30 | False      |
| base_p4_v3                     |              29 |                  30 |        29 | False      |
| base_p5                        |              89 |                  90 |        89 | False      |
| base_p5_v1                     |              15 |                  15 |        15 | False      |
| base_p5_v2                     |               3 |                   3 |         3 | False      |
| base_p5_v3                     |               3 |                   3 |         3 | False      |
| base_p6                        |              87 |                  90 |        87 | False      |
| base_p6_v1                     |               8 |                   8 |         8 | False      |
| base_p6_v2                     |               3 |                   3 |         3 | False      |
| base_p6_v3                     |               3 |                   3 |         3 | False      |
| base_p7                        |              90 |                  90 |        90 | False      |
| base_p7_v1                     |              30 |                  30 |        30 | False      |
| base_p7_v2                     |              30 |                  30 |        30 | False      |
| base_p7_v3                     |              30 |                  30 |        30 | False      |
| base_p8                        |              90 |                  90 |        90 | False      |
| base_p8_v1                     |              18 |                  18 |        18 | False      |
| base_p8_v2                     |               6 |                   6 |         6 | False      |
| base_p8_v3                     |               6 |                   6 |         6 | False      |
| base_p9                        |              90 |                  90 |        90 | False      |
| disentangle_matched_p1         |              29 |                  30 |        29 | False      |
| disentangle_matched_p4         |               9 |                   9 |         9 | False      |
| oracle_t1_p0                   |              30 |                  30 |        30 | False      |
| oracle_t1_p1                   |              30 |                  30 |        30 | False      |
| oracle_t1_p2                   |              30 |                  30 |        30 | False      |
| oracle_t1_p3                   |              30 |                  30 |        30 | False      |
| oracle_t1_p4                   |              30 |                  30 |        30 | False      |
| oracle_t1_p5                   |              30 |                  30 |        30 | False      |
| oracle_t1_p6                   |              30 |                  30 |        30 | False      |
| oracle_t1_p7                   |              30 |                  30 |        30 | False      |
| oracle_t1_p8                   |              30 |                  30 |        30 | False      |
| protocol_a_tavily_p0           |              29 |                  29 |        29 | False      |
| protocol_a_tavily_p1           |              29 |                  29 |        29 | False      |
| protocol_a_tavily_p3           |              28 |                  29 |        28 | False      |
| protocol_a_tavily_p4           |              29 |                  29 |        29 | False      |
| protocol_a_tavily_p5           |              28 |                  29 |        28 | False      |
| protocol_a_tavily_p8           |              29 |                  29 |        29 | False      |


## Missing report files in base patterns (pattern, query_id)

- `base_p11` / `82508e50-497c-445a-b1dd-fd9d7e6dafda`
- `base_p11_16turn` / `ce335c0c-f136-4408-a216-6a891cae861f`
- `base_p11_16turn` / `82508e50-497c-445a-b1dd-fd9d7e6dafda`
- `base_p11_16turn` / `97378684-875a-4a7d-ad19-d7d77942f131`
- `base_p11_16turn` / `e6fbba71-d917-4d1e-bd09-009ff9cd8e9b`
- `base_p11_16turn` / `bea85f62-a722-48a9-8ece-fbfb62653997`
- `base_p11_16turn` / `6b3233e6-9b05-465a-a872-724af25df719`
- `base_p11_16turn` / `ab074bda-d583-42d0-85ce-6c694a51d3ce`
- `base_p11_16turn` / `f6de7687-7cff-4f68-93ea-f632bf6266af`
- `base_p11_16turn` / `8ca825d6-174a-489d-89b9-b000649a2477`
- `base_p11_16turn` / `91408757-a874-44b5-ad5a-66a22b39141d`
- `base_p11_16turn` / `646b575d-383d-483b-82bd-01e5526212c7`
- `base_p11_16turn` / `09456551-1fdc-46fb-a931-4f3a2dfe21a3`
- `base_p11_16turn` / `0a652d00-5c22-4621-8ec4-dd92b1f1450b`
- `base_p11_16turn` / `a4bae04b-4337-4f28-ab8b-5a518cf58fe0`
- `base_p11_16turn` / `f1b0f094-fa7a-4f18-adbd-f4cd86633f77`
- `base_p11_16turn` / `45b0c59c-9d2a-4000-b26e-ee44bf1e7c81`
- `base_p11_16turn` / `c2a1a24b-8a83-453f-9892-d17b533ffe93`
- `base_p11_16turn` / `cc4bd873-0661-4bd2-84bb-928e9e6f1a1c`
- `base_p11_16turn` / `9afd23d2-058d-469f-a663-91286ac0532f`
- `base_p11_16turn` / `adacf6f4-9ec0-4d4e-a8c4-388aad0b2fb0`
- `base_p11_16turn` / `6c59b2a0-c097-4c69-ad03-2c4ea2ec579b`
- `base_p11_16turn` / `ca0edd2d-c9b8-4b85-9b40-754f4865579a`
- `base_p11_16turn` / `1070e6eb-7a8f-4ce9-8818-1931bcfdb2dd`
- `base_p11_16turn` / `49d841f1-0c95-4a87-9b62-5e2a54e8e1a8`
- `base_p11_16turn` / `9f36b55b-28a6-4c4f-b34d-14acc22e5352`
- `base_p11_16turn` / `939cd89c-32e3-4ba3-a9c2-734a9f3b80d1`
- `base_p11_16turn` / `b3c576e7-dfc6-403f-90e7-53c011884d5c`
- `base_p11_16turn` / `36457179-b5e5-4daa-9956-fde342aade9b`
- `base_p11_16turn` / `b02ede76-2353-40f3-9da9-d319c617ab0d`
- `base_p11_16turn` / `eb3965d1-c466-4c4c-bd4a-c8a0b255b893`
- `base_p11_16turn` / `8e99d8d2-f6b9-4800-83a9-6f56829898fe`
- `base_p11_16turn` / `ab17484f-fa6e-4e72-a970-e7cd6c5856d3`
- `base_p11_16turn` / `9db832bc-c38d-4a0b-aebe-ea0fc96da563`
- `base_p11_16turn` / `d1a4c3ea-9507-402c-9ef5-281ea2171c99`
- `base_p11_16turn` / `dsqa_0868`
- `base_p11_16turn` / `dsqa_0788`
- `base_p11_16turn` / `dsqa_0711`
- `base_p11_16turn` / `dsqa_0153`
- `base_p11_16turn` / `dsqa_0403`
- `base_p11_16turn` / `dsqa_0710`
- `base_p11_16turn` / `dsqa_0458`
- `base_p11_16turn` / `dsqa_0148`
- `base_p11_16turn` / `dsqa_0063`
- `base_p11_16turn` / `dsqa_0080`
- `base_p11_16turn` / `dsqa_0125`
- `base_p11_16turn` / `dsqa_0187`
- `base_p11_16turn` / `dsqa_0058`
- `base_p11_16turn` / `174539436421472588-s8`
- `base_p11_16turn` / `174539438444028587-s6`
- `base_p11_16turn` / `174539439157207885-s3`
- `base_p11_16turn` / `174539436825197430-s0`
- `base_p11_16turn` / `174539435410626618-s2`
- `base_p11_16turn` / `174539435612599314-s20`
- `base_p11_16turn` / `174539438140586950-s4`
- `base_p11_16turn` / `174539436120031817-s14`
- `base_p11_16turn` / `2c05315d-6898-4667-b454-d99b7381bedb`
- `base_p11_16turn` / `9f797d29-9f3a-481d-b2fe-326cbc686273`
- `base_p11_16turn` / `26691c84-514b-4712-a43e-09705d681e45`
- `base_p11_16turn` / `653635b7-3bc6-4a7b-98c7-c02038c0e928`
- `base_p11_16turn` / `10cece36-a507-4a93-9600-13f3e0e677f8`
- `base_p11_16turn` / `c6f097c9-2216-4e98-af45-8101681b38ec`
- `base_p1_7b` / `ce335c0c-f136-4408-a216-6a891cae861f`
- `base_p1_7b` / `82508e50-497c-445a-b1dd-fd9d7e6dafda`
- `base_p1_7b` / `e15a37b4-cd7f-4f52-ba81-9f33b24aabe9`
- `base_p1_7b` / `35bc9c48-0853-469d-8bf7-8f6947ec4fb7`
- `base_p1_7b` / `97378684-875a-4a7d-ad19-d7d77942f131`
- `base_p1_7b` / `e6fbba71-d917-4d1e-bd09-009ff9cd8e9b`
- `base_p1_7b` / `bea85f62-a722-48a9-8ece-fbfb62653997`
- `base_p1_7b` / `6b3233e6-9b05-465a-a872-724af25df719`
- `base_p1_7b` / `ab074bda-d583-42d0-85ce-6c694a51d3ce`
- `base_p1_7b` / `f6de7687-7cff-4f68-93ea-f632bf6266af`
- `base_p1_7b` / `8ca825d6-174a-489d-89b9-b000649a2477`
- `base_p1_7b` / `91408757-a874-44b5-ad5a-66a22b39141d`
- `base_p1_7b` / `c63923e7-6ab4-4742-b739-49f5d1f1cfe4`
- `base_p1_7b` / `646b575d-383d-483b-82bd-01e5526212c7`
- `base_p1_7b` / `09456551-1fdc-46fb-a931-4f3a2dfe21a3`
- `base_p1_7b` / `0a652d00-5c22-4621-8ec4-dd92b1f1450b`
- `base_p1_7b` / `a4bae04b-4337-4f28-ab8b-5a518cf58fe0`
- `base_p1_7b` / `f1b0f094-fa7a-4f18-adbd-f4cd86633f77`
- `base_p1_7b` / `45b0c59c-9d2a-4000-b26e-ee44bf1e7c81`
- `base_p1_7b` / `c2a1a24b-8a83-453f-9892-d17b533ffe93`
- `base_p1_7b` / `cc4bd873-0661-4bd2-84bb-928e9e6f1a1c`
- `base_p1_7b` / `9afd23d2-058d-469f-a663-91286ac0532f`
- `base_p1_7b` / `4f36f602-6c14-4c23-a37d-e5ee92ed4552`
- `base_p1_7b` / `adacf6f4-9ec0-4d4e-a8c4-388aad0b2fb0`
- `base_p1_7b` / `6c59b2a0-c097-4c69-ad03-2c4ea2ec579b`
- `base_p1_7b` / `ca0edd2d-c9b8-4b85-9b40-754f4865579a`
- `base_p1_7b` / `1070e6eb-7a8f-4ce9-8818-1931bcfdb2dd`
- `base_p1_7b` / `49d841f1-0c95-4a87-9b62-5e2a54e8e1a8`
- `base_p1_7b` / `9f36b55b-28a6-4c4f-b34d-14acc22e5352`
- `base_p1_7b` / `939cd89c-32e3-4ba3-a9c2-734a9f3b80d1`
- `base_p1_7b` / `b3c576e7-dfc6-403f-90e7-53c011884d5c`
- `base_p1_7b` / `aca95495-b0fe-4330-90fc-d2fd1e2c709e`
- `base_p1_7b` / `36457179-b5e5-4daa-9956-fde342aade9b`
- `base_p1_7b` / `eb3965d1-c466-4c4c-bd4a-c8a0b255b893`
- `base_p1_7b` / `8e99d8d2-f6b9-4800-83a9-6f56829898fe`
- `base_p1_7b` / `ab17484f-fa6e-4e72-a970-e7cd6c5856d3`
- `base_p1_7b` / `9db832bc-c38d-4a0b-aebe-ea0fc96da563`
- `base_p1_7b` / `d1a4c3ea-9507-402c-9ef5-281ea2171c99`
- `base_p1_7b` / `dsqa_0868`
- `base_p1_7b` / `dsqa_0002`
- `base_p1_7b` / `dsqa_0788`
- `base_p1_7b` / `dsqa_0711`
- `base_p1_7b` / `dsqa_0153`
- `base_p1_7b` / `dsqa_0403`
- `base_p1_7b` / `dsqa_0593`
- `base_p1_7b` / `dsqa_0109`
- `base_p1_7b` / `dsqa_0710`
- `base_p1_7b` / `dsqa_0458`
- `base_p1_7b` / `dsqa_0148`
- `base_p1_7b` / `dsqa_0063`
- `base_p1_7b` / `dsqa_0080`
- `base_p1_7b` / `dsqa_0125`
- `base_p1_7b` / `dsqa_0187`
- `base_p1_7b` / `dsqa_0058`
- `base_p1_7b` / `dsqa_0892`
- `base_p1_7b` / `174539438139972353-s34`
- `base_p1_7b` / `174539436421472588-s8`
- `base_p1_7b` / `174539439358887410-s1`
- `base_p1_7b` / `174539438444028587-s6`
- `base_p1_7b` / `174539439157207885-s3`
- `base_p1_7b` / `174539437127211049-s8`
- `base_p1_7b` / `174539436825197430-s0`
- `base_p1_7b` / `174539435410626618-s2`
- `base_p1_7b` / `174539435612599314-s20`
- `base_p1_7b` / `174539438140586950-s4`
- `base_p1_7b` / `174539437936871188-s9`
- `base_p1_7b` / `174539438039017258-s17`
- `base_p1_7b` / `174539436120031817-s14`
- `base_p1_7b` / `a45c277e-55d9-4e7f-b1de-37fc2e19daf6`
- `base_p1_7b` / `82de3e92-abe2-46ac-ad17-23417b9c4da7`
- `base_p1_7b` / `2c05315d-6898-4667-b454-d99b7381bedb`
- `base_p1_7b` / `9f797d29-9f3a-481d-b2fe-326cbc686273`
- `base_p1_7b` / `26691c84-514b-4712-a43e-09705d681e45`
- `base_p1_7b` / `653635b7-3bc6-4a7b-98c7-c02038c0e928`
- `base_p1_7b` / `6aa10957-bdd9-4dab-a4e1-234a17cb87dd`
- `base_p1_7b` / `10cece36-a507-4a93-9600-13f3e0e677f8`
- `base_p1_7b` / `c6f097c9-2216-4e98-af45-8101681b38ec`
- `base_p1_7b` / `d65103ae-c881-4116-a0a7-1b233eb6275a`
- `base_p3` / `82508e50-497c-445a-b1dd-fd9d7e6dafda`
- `base_p4_7b` / `ce335c0c-f136-4408-a216-6a891cae861f`
- `base_p4_7b` / `82508e50-497c-445a-b1dd-fd9d7e6dafda`
- `base_p4_7b` / `e15a37b4-cd7f-4f52-ba81-9f33b24aabe9`
- `base_p4_7b` / `35bc9c48-0853-469d-8bf7-8f6947ec4fb7`
- `base_p4_7b` / `97378684-875a-4a7d-ad19-d7d77942f131`
- `base_p4_7b` / `e6fbba71-d917-4d1e-bd09-009ff9cd8e9b`
- `base_p4_7b` / `bea85f62-a722-48a9-8ece-fbfb62653997`
- `base_p4_7b` / `6b3233e6-9b05-465a-a872-724af25df719`
- `base_p4_7b` / `ab074bda-d583-42d0-85ce-6c694a51d3ce`
- `base_p4_7b` / `f6de7687-7cff-4f68-93ea-f632bf6266af`
- `base_p4_7b` / `8ca825d6-174a-489d-89b9-b000649a2477`
- `base_p4_7b` / `91408757-a874-44b5-ad5a-66a22b39141d`
- `base_p4_7b` / `c63923e7-6ab4-4742-b739-49f5d1f1cfe4`
- `base_p4_7b` / `646b575d-383d-483b-82bd-01e5526212c7`
- `base_p4_7b` / `09456551-1fdc-46fb-a931-4f3a2dfe21a3`
- `base_p4_7b` / `0a652d00-5c22-4621-8ec4-dd92b1f1450b`
- `base_p4_7b` / `a4bae04b-4337-4f28-ab8b-5a518cf58fe0`
- `base_p4_7b` / `f1b0f094-fa7a-4f18-adbd-f4cd86633f77`
- `base_p4_7b` / `45b0c59c-9d2a-4000-b26e-ee44bf1e7c81`
- `base_p4_7b` / `c2a1a24b-8a83-453f-9892-d17b533ffe93`
- `base_p4_7b` / `cc4bd873-0661-4bd2-84bb-928e9e6f1a1c`
- `base_p4_7b` / `9afd23d2-058d-469f-a663-91286ac0532f`
- `base_p4_7b` / `4f36f602-6c14-4c23-a37d-e5ee92ed4552`
- `base_p4_7b` / `adacf6f4-9ec0-4d4e-a8c4-388aad0b2fb0`
- `base_p4_7b` / `6c59b2a0-c097-4c69-ad03-2c4ea2ec579b`
- `base_p4_7b` / `ca0edd2d-c9b8-4b85-9b40-754f4865579a`
- `base_p4_7b` / `1070e6eb-7a8f-4ce9-8818-1931bcfdb2dd`
- `base_p4_7b` / `49d841f1-0c95-4a87-9b62-5e2a54e8e1a8`
- `base_p4_7b` / `9f36b55b-28a6-4c4f-b34d-14acc22e5352`
- `base_p4_7b` / `939cd89c-32e3-4ba3-a9c2-734a9f3b80d1`
- `base_p4_7b` / `b3c576e7-dfc6-403f-90e7-53c011884d5c`
- `base_p4_7b` / `aca95495-b0fe-4330-90fc-d2fd1e2c709e`
- `base_p4_7b` / `36457179-b5e5-4daa-9956-fde342aade9b`
- `base_p4_7b` / `eb3965d1-c466-4c4c-bd4a-c8a0b255b893`
- `base_p4_7b` / `8e99d8d2-f6b9-4800-83a9-6f56829898fe`
- `base_p4_7b` / `ab17484f-fa6e-4e72-a970-e7cd6c5856d3`
- `base_p4_7b` / `9db832bc-c38d-4a0b-aebe-ea0fc96da563`
- `base_p4_7b` / `d1a4c3ea-9507-402c-9ef5-281ea2171c99`
- `base_p4_7b` / `dsqa_0868`
- `base_p4_7b` / `dsqa_0002`
- `base_p4_7b` / `dsqa_0788`
- `base_p4_7b` / `dsqa_0711`
- `base_p4_7b` / `dsqa_0153`
- `base_p4_7b` / `dsqa_0403`
- `base_p4_7b` / `dsqa_0593`
- `base_p4_7b` / `dsqa_0109`
- `base_p4_7b` / `dsqa_0710`
- `base_p4_7b` / `dsqa_0458`
- `base_p4_7b` / `dsqa_0148`
- `base_p4_7b` / `dsqa_0063`
- `base_p4_7b` / `dsqa_0080`
- `base_p4_7b` / `dsqa_0125`
- `base_p4_7b` / `dsqa_0187`
- `base_p4_7b` / `dsqa_0058`
- `base_p4_7b` / `dsqa_0892`
- `base_p4_7b` / `174539438139972353-s34`
- `base_p4_7b` / `174539436421472588-s8`
- `base_p4_7b` / `174539439358887410-s1`
- `base_p4_7b` / `174539438444028587-s6`
- `base_p4_7b` / `174539439157207885-s3`
- `base_p4_7b` / `174539437127211049-s8`
- `base_p4_7b` / `174539436825197430-s0`
- `base_p4_7b` / `174539435410626618-s2`
- `base_p4_7b` / `174539435612599314-s20`
- `base_p4_7b` / `174539438140586950-s4`
- `base_p4_7b` / `174539437936871188-s9`
- `base_p4_7b` / `174539438039017258-s17`
- `base_p4_7b` / `174539436120031817-s14`
- `base_p4_7b` / `a45c277e-55d9-4e7f-b1de-37fc2e19daf6`
- `base_p4_7b` / `82de3e92-abe2-46ac-ad17-23417b9c4da7`
- `base_p4_7b` / `2c05315d-6898-4667-b454-d99b7381bedb`
- `base_p4_7b` / `9f797d29-9f3a-481d-b2fe-326cbc686273`
- `base_p4_7b` / `26691c84-514b-4712-a43e-09705d681e45`
- `base_p4_7b` / `653635b7-3bc6-4a7b-98c7-c02038c0e928`
- `base_p4_7b` / `6aa10957-bdd9-4dab-a4e1-234a17cb87dd`
- `base_p4_7b` / `10cece36-a507-4a93-9600-13f3e0e677f8`
- `base_p4_7b` / `c6f097c9-2216-4e98-af45-8101681b38ec`
- `base_p4_7b` / `d65103ae-c881-4116-a0a7-1b233eb6275a`
- `base_p5` / `82508e50-497c-445a-b1dd-fd9d7e6dafda`
- `base_p6` / `82508e50-497c-445a-b1dd-fd9d7e6dafda`
- `base_p6` / `0a652d00-5c22-4621-8ec4-dd92b1f1450b`
- `base_p6` / `dsqa_0683`


## Ablation coverage notes

- **ablation_p3_no_quality_eval**: reports=89/90; judge coverage={'claude_opus': 3, 'claude_sonnet': 87, 'gpt52': 87}
- **ablation_p3_no_topic_mining**: reports=89/90; judge coverage={'claude_sonnet': 86, 'gpt52': 86}
- **ablation_p4_fixed_perspectives**: reports=90/90; judge coverage={'claude_sonnet': 90, 'gpt52': 90}
- **ablation_p4_no_conversations**: reports=90/90; judge coverage={'claude_opus': 8, 'claude_sonnet': 90, 'gpt52': 90}
- **ablation_p4_no_triangulation**: reports=90/90; judge coverage={'claude_sonnet': 90, 'gpt52': 89}
- **ablation_p5_fixed_width**: reports=90/90; judge coverage={'claude_sonnet': 82, 'gpt52': 82}
- **ablation_p5_no_citation_verify** [EXCLUDED]: reports=54/90; judge coverage={'claude_sonnet': 2}
- **ablation_p5_no_meta_eval**: reports=90/90; judge coverage={'claude_sonnet': 86, 'gpt52': 86}

Exclusion criterion: Only 2/90 reports were generated for ablation_p5_no_citation_verify (run aborted early); statistical comparisons would be unreliable.


## Overall score verification (stored vs recomputed)

- `overall_score_recomputed` uses source-type weights from `DIMENSION_WEIGHTS_BY_SOURCE` (matches the gpt52 stored convention).
- `overall_score_per_query_weights` uses per-query `rubric.dimension_weights` from the eval manifest.
- `overall_score_trustworthy` is False for claude_sonnet (upstream-corrupted).
- Rows compared: **6509**
- Rows with |stored - recomputed| > 0.01: **1045**

Per-judge stored-vs-recomputed delta:

| judge         |       mean |   median |      max |
|:--------------|-----------:|---------:|---------:|
| claude_code   | 0.00572197 |  3.5e-05 | 0.050035 |
| claude_opus   | 0.00585893 |  3.5e-05 | 0.10914  |
| claude_sonnet | 0.0486649  |  0.0125  | 0.3241   |
| gpt52         | 0.00357708 |  3.5e-05 | 0.050035 |


## Verdict satisfied distribution per judge

| judge         |   n_verdicts |   n_satisfied |   satisfaction_rate |
|:--------------|-------------:|--------------:|--------------------:|
| claude_code   |        10600 |          5471 |            0.516132 |
| claude_opus   |        45666 |         28755 |            0.629681 |
| claude_sonnet |        75377 |         54256 |            0.719795 |
| gpt52         |       116893 |         52123 |            0.445904 |


## Total tokens per pattern (summary stats)

| pattern                        |   count |             mean |      std |    min |           median |              max |
|:-------------------------------|--------:|-----------------:|---------:|-------:|-----------------:|-----------------:|
| ablation_p3_no_quality_eval    |      90 | 298385           | 133122   |      0 | 289567           | 618266           |
| ablation_p3_no_topic_mining    |      90 | 271463           | 124174   |      0 | 264752           | 793333           |
| ablation_p4_fixed_perspectives |      90 | 438494           | 149874   | 112061 | 432056           | 838229           |
| ablation_p4_no_conversations   |      90 | 309247           | 235819   |  64058 | 228876           |      1.22032e+06 |
| ablation_p4_no_triangulation   |      90 |      1.05864e+06 | 411680   | 218977 |      1.08118e+06 |      2.09283e+06 |
| ablation_p5_fixed_width        |      90 | 705340           | 262022   | 113617 | 674172           |      1.61572e+06 |
| ablation_p5_no_citation_verify |      55 | 664482           | 256983   |      0 | 646936           |      1.27452e+06 |
| ablation_p5_no_meta_eval       |      90 | 379466           | 175363   |  61499 | 345235           | 843467           |
| base_p0                        |      90 |  64691.6         |  41765.3 |      0 |  59760.5         | 184580           |
| base_p0_v1                     |      30 |  80370.6         |  42493.6 |   2164 |  74920           | 182956           |
| base_p0_v10                    |      30 |  80725.3         |  41597.8 |   2164 |  75768.5         | 183535           |
| base_p0_v11                    |      30 |  80949.6         |  41590.3 |   2164 |  76554           | 182553           |
| base_p0_v2                     |      30 |  80615.1         |  41936.8 |   2164 |  76140           | 183257           |
| base_p0_v3                     |      30 |  79940.8         |  42112.4 |   2164 |  74687           | 184011           |
| base_p0_v4                     |      30 |  80868.2         |  41704.2 |   2164 |  75437.5         | 183756           |
| base_p0_v5                     |      30 |  80962.6         |  41779.5 |   2164 |  75417           | 183676           |
| base_p0_v6                     |      30 |  80878.3         |  41256.8 |   2164 |  75642.5         | 182780           |
| base_p0_v7                     |      30 |  80997.2         |  41968.7 |   2164 |  76792           | 184044           |
| base_p0_v8                     |      30 |  80604.7         |  41567.4 |   2164 |  75953           | 182922           |
| base_p0_v9                     |      30 |  81057.3         |  41543.7 |   2164 |  75353           | 183075           |
| base_p1                        |      90 | 781068           | 316050   | 164014 | 798944           |      1.75524e+06 |
| base_p10                       |      89 | 348658           | 326186   |  32215 | 212408           |      1.90915e+06 |
| base_p10_v1                    |      30 | 332391           | 252714   |  77005 | 245648           |      1.22858e+06 |
| base_p10_v2                    |      30 | 332152           | 252420   |  77005 | 245648           |      1.22858e+06 |
| base_p10_v3                    |      30 | 332142           | 252425   |  77005 | 245648           |      1.22858e+06 |
| base_p11                       |      89 |  48343.7         |  38867.7 |   6139 |  36354           | 171826           |
| base_p11_16turn                |      29 |  67549.2         |  42309.6 |  11148 |  52130           | 187682           |
| base_p12                       |      90 |  55095.5         |  44364.2 |      0 |  42179.5         | 200661           |
| base_p1_7b                     |      12 | 143768           |  68183.3 |  39628 | 126406           | 281391           |
| base_p1_v1                     |      30 | 364861           | 193008   |  71505 | 300193           | 758842           |
| base_p1_v2                     |      30 | 366706           | 177398   |  88444 | 330516           | 768693           |
| base_p1_v3                     |      30 | 383586           | 185367   | 105562 | 362824           | 910278           |
| base_p2                        |      90 | 182061           |  96450.3 |  33455 | 159670           | 512262           |
| base_p3                        |      89 | 316073           | 208432   |  98613 | 254451           |      1.15685e+06 |
| base_p4                        |      90 |      1.13221e+06 | 537007   |  92386 |      1.15632e+06 |      2.8232e+06  |
| base_p4_7b                     |      12 | 677952           | 395845   | 240079 | 577029           |      1.50914e+06 |
| base_p4_v1                     |      30 |      1.19485e+06 | 588843   | 363041 |      1.08856e+06 |      3.05058e+06 |
| base_p4_v2                     |      30 |      1.15507e+06 | 563962   | 278707 |      1.00756e+06 |      2.52757e+06 |
| base_p4_v3                     |      29 |      1.10117e+06 | 491657   | 253339 |      1.06423e+06 |      2.36244e+06 |
| base_p5                        |      89 | 714602           | 279454   |  28057 | 678391           |      1.67301e+06 |
| base_p5_v1                     |      15 | 668110           | 266711   | 137353 | 638599           |      1.06634e+06 |
| base_p5_v2                     |       3 | 988730           | 164487   | 851490 | 943640           |      1.17106e+06 |
| base_p5_v3                     |       3 | 963425           | 256645   | 671649 |      1.0644e+06  |      1.15422e+06 |
| base_p6                        |      87 | 529175           | 371716   |  54518 | 395137           |      1.9226e+06  |
| base_p6_v1                     |       8 | 640138           | 243305   | 291748 | 708164           | 902351           |
| base_p6_v2                     |       3 | 749457           | 381497   | 377303 | 731412           |      1.13966e+06 |
| base_p6_v3                     |       3 | 642028           | 500750   | 260402 | 456647           |      1.20903e+06 |
| base_p7                        |      90 | 485001           | 228211   | 106018 | 459897           |      1.62622e+06 |
| base_p7_v1                     |      30 | 524333           | 311506   | 126482 | 459702           |      1.70004e+06 |
| base_p7_v2                     |      30 | 522171           | 272174   | 142445 | 420106           |      1.11213e+06 |
| base_p7_v3                     |      30 | 485803           | 266395   | 175368 | 423720           |      1.21729e+06 |
| base_p8                        |      90 | 579816           | 176156   | 249757 | 595170           |      1.09785e+06 |
| base_p8_v1                     |      18 | 568658           | 153902   | 307739 | 530678           | 827455           |
| base_p8_v2                     |       6 | 610903           | 270003   | 379283 | 532766           |      1.12761e+06 |
| base_p8_v3                     |       6 | 594064           | 119118   | 478758 | 589002           | 804316           |
| base_p9                        |      90 |  54817           |  44605.6 |      0 |  42810.5         | 200422           |
| disentangle_matched_p1         |      29 | 204420           |  97904.6 |  22363 | 190856           | 352664           |
| disentangle_matched_p4         |       9 | 187353           | 106403   |  51354 | 186758           | 388140           |
| oracle_t1_p0                   |      30 | 111300           |  37539   |  36066 | 127204           | 162362           |
| oracle_t1_p1                   |      30 | 165861           |  53457.5 |  59285 | 177654           | 230507           |
| oracle_t1_p2                   |      30 | 498517           | 145522   | 200347 | 511271           | 932325           |
| oracle_t1_p3                   |      30 | 136711           |  46521.8 |  46461 | 154684           | 200590           |
| oracle_t1_p4                   |      30 |      1.20851e+06 | 384024   | 343596 |      1.29787e+06 |      1.72179e+06 |
| oracle_t1_p5                   |      30 | 924617           | 242518   | 396683 | 984336           |      1.23316e+06 |
| oracle_t1_p6                   |      30 | 207671           |  48511.9 | 154557 | 187656           | 339507           |
| oracle_t1_p7                   |      30 |      2.1038e+06  | 516613   | 904626 |      2.12378e+06 |      2.96991e+06 |
| oracle_t1_p8                   |      30 | 748916           | 134481   | 374492 | 764328           |      1.0581e+06  |
| protocol_a_tavily_p0           |      29 |  39329.8         |  54890.1 |      0 |   3006           | 172960           |
| protocol_a_tavily_p1           |      29 |  65823.2         |  38228.4 |  25381 |  53196           | 167177           |
| protocol_a_tavily_p3           |      28 |   9690.21        |  11039.5 |   4673 |   5667.5         |  57195           |
| protocol_a_tavily_p4           |      29 | 266249           |  70763.5 | 201923 | 231566           | 435331           |
| protocol_a_tavily_p5           |      28 |  87471.5         |  93116.8 |  11672 |  33063           | 362095           |
| protocol_a_tavily_p8           |      29 |  45061.7         |  27846.6 |  16636 |  35791           | 125936           |


## Cost proxy (USD) per pattern

| pattern                        |   count |       mean |      median |       sum |
|:-------------------------------|--------:|-----------:|------------:|----------:|
| ablation_p3_no_quality_eval    |      90 |  1.49192   |  1.44783    | 134.273   |
| ablation_p3_no_topic_mining    |      90 |  1.35732   |  1.32376    | 122.159   |
| ablation_p4_fixed_perspectives |      90 |  2.19247   |  2.16028    | 197.322   |
| ablation_p4_no_conversations   |      90 |  1.54624   |  1.14438    | 139.161   |
| ablation_p4_no_triangulation   |      90 |  5.29319   |  5.40588    | 476.387   |
| ablation_p5_fixed_width        |      90 |  3.5267    |  3.37086    | 317.403   |
| ablation_p5_no_citation_verify |      55 |  3.32241   |  3.23468    | 182.733   |
| ablation_p5_no_meta_eval       |      90 |  1.89733   |  1.72618    | 170.76    |
| base_p0                        |      90 |  0.323458  |  0.298802   |  29.1112  |
| base_p0_v1                     |      30 |  0.401853  |  0.3746     |  12.0556  |
| base_p0_v10                    |      30 |  0.403627  |  0.378842   |  12.1088  |
| base_p0_v11                    |      30 |  0.404748  |  0.38277    |  12.1424  |
| base_p0_v2                     |      30 |  0.403075  |  0.3807     |  12.0923  |
| base_p0_v3                     |      30 |  0.399704  |  0.373435   |  11.9911  |
| base_p0_v4                     |      30 |  0.404341  |  0.377188   |  12.1302  |
| base_p0_v5                     |      30 |  0.404813  |  0.377085   |  12.1444  |
| base_p0_v6                     |      30 |  0.404392  |  0.378212   |  12.1317  |
| base_p0_v7                     |      30 |  0.404986  |  0.38396    |  12.1496  |
| base_p0_v8                     |      30 |  0.403023  |  0.379765   |  12.0907  |
| base_p0_v9                     |      30 |  0.405286  |  0.376765   |  12.1586  |
| base_p1                        |      90 |  3.90534   |  3.99472    | 351.481   |
| base_p10                       |      89 |  0.0653057 |  0.0521268  |   5.81221 |
| base_p10_v1                    |      30 |  0.0406682 |  0.0408908  |   1.22005 |
| base_p10_v2                    |      30 |  0.0413122 |  0.0422989  |   1.23937 |
| base_p10_v3                    |      30 |  0.0401258 |  0.0403858  |   1.20377 |
| base_p11                       |      89 |  0.241718  |  0.18177    |  21.5129  |
| base_p11_16turn                |      29 |  0.337746  |  0.26065    |   9.79463 |
| base_p12                       |      90 |  0.0165148 |  0.0162243  |   1.48633 |
| base_p1_7b                     |      12 |  0.718839  |  0.632027   |   8.62607 |
| base_p1_v1                     |      30 |  1.82431   |  1.50096    |  54.7292  |
| base_p1_v2                     |      30 |  1.83353   |  1.65258    |  55.0058  |
| base_p1_v3                     |      30 |  1.91793   |  1.81412    |  57.5379  |
| base_p2                        |      90 |  0.910305  |  0.79835    |  81.9275  |
| base_p3                        |      89 |  1.58037   |  1.27225    | 140.653   |
| base_p4                        |      90 |  5.66106   |  5.78161    | 509.496   |
| base_p4_7b                     |      12 |  3.38976   |  2.88514    |  40.6771  |
| base_p4_v1                     |      30 |  5.97424   |  5.4428     | 179.227   |
| base_p4_v2                     |      30 |  5.77533   |  5.03779    | 173.26    |
| base_p4_v3                     |      29 |  5.50587   |  5.32113    | 159.67    |
| base_p5                        |      89 |  3.57301   |  3.39195    | 317.998   |
| base_p5_v1                     |      15 |  3.34055   |  3.19299    |  50.1083  |
| base_p5_v2                     |       3 |  4.94365   |  4.7182     |  14.831   |
| base_p5_v3                     |       3 |  4.81712   |  5.32201    |  14.4514  |
| base_p6                        |      87 |  2.64587   |  1.97568    | 230.191   |
| base_p6_v1                     |       8 |  3.20069   |  3.54082    |  25.6055  |
| base_p6_v2                     |       3 |  3.74729   |  3.65706    |  11.2419  |
| base_p6_v3                     |       3 |  3.21014   |  2.28323    |   9.63041 |
| base_p7                        |      90 |  2.425     |  2.29948    | 218.25    |
| base_p7_v1                     |      30 |  2.62167   |  2.29851    |  78.65    |
| base_p7_v2                     |      30 |  2.61086   |  2.10053    |  78.3257  |
| base_p7_v3                     |      30 |  2.42902   |  2.1186     |  72.8705  |
| base_p8                        |      90 |  2.89908   |  2.97585    | 260.917   |
| base_p8_v1                     |      18 |  2.84329   |  2.65339    |  51.1793  |
| base_p8_v2                     |       6 |  3.05452   |  2.66383    |  18.3271  |
| base_p8_v3                     |       6 |  2.97032   |  2.94501    |  17.8219  |
| base_p9                        |      90 |  0.0115734 |  0.00912143 |   1.04161 |
| disentangle_matched_p1         |      29 |  1.0221    |  0.95428    |  29.6409  |
| disentangle_matched_p4         |       9 |  0.936765  |  0.93379    |   8.43088 |
| oracle_t1_p0                   |      30 |  0.556499  |  0.636023   |  16.695   |
| oracle_t1_p1                   |      30 |  0.829305  |  0.888273   |  24.8791  |
| oracle_t1_p2                   |      30 |  2.49258   |  2.55635    |  74.7775  |
| oracle_t1_p3                   |      30 |  0.683555  |  0.773418   |  20.5067  |
| oracle_t1_p4                   |      30 |  6.04257   |  6.48936    | 181.277   |
| oracle_t1_p5                   |      30 |  4.62308   |  4.92168    | 138.692   |
| oracle_t1_p6                   |      30 |  1.03836   |  0.938283   |  31.1507  |
| oracle_t1_p7                   |      30 | 10.519     | 10.6189     | 315.569   |
| oracle_t1_p8                   |      30 |  3.74458   |  3.82164    | 112.337   |
| protocol_a_tavily_p0           |      29 |  0.196649  |  0.01503    |   5.70282 |
| protocol_a_tavily_p1           |      29 |  0.329116  |  0.26598    |   9.54437 |
| protocol_a_tavily_p3           |      28 |  0.0484511 |  0.0283375  |   1.35663 |
| protocol_a_tavily_p4           |      29 |  1.33125   |  1.15783    |  38.6061  |
| protocol_a_tavily_p5           |      28 |  0.437358  |  0.165315   |  12.246   |
| protocol_a_tavily_p8           |      29 |  0.225309  |  0.178955   |   6.53395 |


## Schema anomalies and notes

- 4 criterion_id(s) mapped to >1 dimension across judges (rubric drift warning)
- 349 (pattern, query_id, dimension) groups show different criterion sets across judges — rubric version drift

## Dataframe row counts

- df_queries: 90
- df_runs: 6570
- df_scores: 58552
- df_overall_scores: 6509
- df_verdicts: 248536
