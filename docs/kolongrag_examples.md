# Ko-LongRAG Examples
[ Single-doc QA Answerable Case ]
Context:
...
Title: 박진우 (야구인)
Text: 박진우(朴晋佑, 1990년 2월 12일 ∼ )는 전 KBO 리그 NC 다이노스의 투수이자, 현 KBO리그 SSG 랜더스의 스카우트이다.
...
2019년 시즌 : 선발과 불펜을 가리지 않고 활약했다. 시즌 140.2이닝 3점대 평균자책점, 92탈삼진, 9승 7패, 5홀드를 기록했다. 이동욱 감독은 ’가장 MVP로 꼽고 싶은 선수’라며칭찬했다.
...
Question: 박진우가 NC 다이노스에서 9승을 기록한 시즌은 언제인가요?
Answer: 2019년
[ Single-doc QA Unanswerable Case ]
Question: 인천남동소방서의 설립 연도는 무엇인가요?
Answer: 주어진 문서내에서 답할 수 있는 정보가 충분하지 않습니다.
[ Multi-doc QA Answerable Case ]
Context:
...
Title: 아스투리아스 공상
Text: 아스투리아스 공상은 스페인의 프린시페 데 아스투리아스 재단(Fundación Príncipe de Asturias)이 주관하는 상이다. 1980년 9월 24일 스페인의 왕세자에 해당하는 호칭인아스투리아스 공이었던 펠리페 (Felipe, 펠리페 6세)에 의해 제정되었으며 1981년에 첫시상식이 열렸다. 총 9개 부문 (예술 부문, 커뮤니케이션 · 인문주의 부문, 국제 협력 부문,문학 부문, 사회과학 부문, 체육 부문, 기술 · 과학 연구 부문, 화합 부문, 아스투리아스 모범상부문)으로 나누어 시상한다. 시상식은 아스투리아스 지방의 오비에도에서 열린다. 수상자는주안 미로가 제작한 조각, 상금 50,000 유로를 받게 된다.
...
Title: 미겔 데 세르반테스 상
Text: 미겔 데 세르반테스 상(-賞, ) 또는 세르반테스 상은 스페인 작가 미겔 데 세르반테스의이름이 붙은 스페인어 작가에게 수여되는 문학상으로, 영연방의 맨 부커 상과 유사한스페인어권의 상이다. 그러나 맨 부커 상과는 다르게 일생 동안의 문학적 성취를 평가해서 단한 번만 수여하므로 스페인어권에서 그 권위는 노벨 문학상에 버금간다. 1976년 제정되었다.스페인 문화부가 수여하며 상금은 12만 5천 유로이다.
...
Question: 아스투리아스 공상과 미겔 데 세르반테스 상 중 상금이 더 많은 것은 무엇인가요?Answer: 미겔 데 세르반테스 상
[ Multi-doc QA Unanswerable Case ]
Question: 넬슨 록펠러와 노아 사이러스는 둘 다 정치 경력을 가지고 있었나요?
Answer: 주어진 문서내에서 답할 수 있는 정보가 충분하지 않습니다.

# Ko-LongRAG Prompt
[ Single-doc QA Prompt ]
System: 당신은 도움이 되는 어시스턴트입니다.
User:
다음 문서를 살펴보고, 질문에 대한 답을 추출하세요.
질문에 대한 답만 생성하세요. 답변은 매우 간결해야 합니다.
답변을 문서에서 찾을 수 없는 경우,
’주어진 정보로 답할 수 없다’로 응답하세요.
문서는 “Title”에 따라 정렬된 Wikipedia 문단 목록이며 제목은 다음과 같습니다: {{titles}}.각 위키피디아 문단은 ’Title’ 필드와 ’Text’ 필드를 포함합니다.
문서는 다음과 같습니다: {{context}}. 질문은 다음과 같습니다: {{question}}.
[ Multi-doc QA Prompt ]
System: 당신은 도움이 되는 어시스턴트입니다.
User:
다음 문서를 검토하고 질문에 답하세요.
문서는 Wikipedia 문단 목록이며 제목은 다음과 같습니다: {{titles}}.
질문에는 두 가지 유형이 있습니다:
예 또는 아니오로 답하거나 두 후보 중에서 선택해야 하는 비교 질문과, 단답형 형태의 일반질문입니다.
문서는 다음과 같습니다: {{context}}.
문서에서 필요한 문단을 찾아 질문에 답하세요: {{question}}.
일반 질문의 경우 문단에서 정확한 단어를 찾아서 답변해야 합니다.
질문에 대한 답만 생성하고 다른 어떤 것도 생성하지 마세요.
답변을 문서에서 찾을 수 없는 경우, ’주어진 정보로 답할 수 없다’로 응답하세요.

# Ko-LongRAG LLM-as-a-Judge Prompt
System:
You are an expert evaluator of text answers in Korean.
Your task is to compare the content of two Korean answers, a long answer (long_ans) and a short answer (short_ans), with the provided correct answers (Answer), which may contain multiple correct options.
Both the long answer and the short answer need to be checked for correctness. The long and short answers do not need to match any of the answers in the Answer list word-for-word but must convey the same key meaning or idea.
If either the long or short answer matches any one of the correct answers in the Answer list, it should be considered correct.
Focus only on the accuracy of the content and ignore style, tone, or extra information unless it introduces inaccuracies.
For both the long and short answers, return only the evaluation result as a Python dictionary object, and ensure the output is formatted as valid Python code.
Here are two examples of how to evaluate answers:
Example 1:
Question: HP는 게임에서 무엇을 의미하나요?
Answer: ['체력', '생명력']
long_ans: HP는 ’생명력’ 또는 ’체력’을 의미하며, 게임에서 캐릭터의 생존력을 나타내는지표입니다. HP가 줄어들면 캐릭터는 점점 약해지며, 0이 되면 게임에서 탈락하거나 패배할수 있습니다.
short_ans: HP는 캐릭터의 체력입니다.
Evaluation: {'long_ans': 'correct', 'short_ans': 'correct'}
Example 2:
Question: 프랑스의 수도는 어디인가요?
Answer: ['파리']
long_ans: 프랑스의 수도는 파리로, 리옹의 오른쪽 아래에 위치하고, 문화와 예술의 중심지로알려져 있습니다. 에펠탑, 루브르 박물관, 노트르담 대성당 등 유명한 관광지가 위치해있습니다.
short_ans: 프랑스의 수도는 리옹입니다.
Evaluation: {'long_ans': 'correct', 'short_ans': 'incorrect'}
Now, proceed with your evaluation of the following question, answer, and responses, and return only the evaluation as a valid Python dictionary.
Ensure the response is a valid Python dictionary object without any additional text.
User:
Evaluate the following long and short answers based on the provided correct answer.
Your goal is to determine if the long and short answers are correct.
Return the evaluation result in the form of a Python dictionary: {'long_ans': 'correct 'or 'incorrect ', 'short_ans': 'correct'or 'incorrect'}.
Question: {{question}}
Answer: {{answer}}
long_ans: {{long_ans}}
short_ans: {{short_ans}}
Return only the evaluation in the form of a Python dictionary.
Do not include any explanation or additional comments.