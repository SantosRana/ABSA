def build_prompt(review_text):

    return f"""
You are an expert at aspect-based sentiment analysis.

Task:
Analyze a Korean restaurant review and classify the sentiment for each of the four aspect
below.

Aspects:
1. FOOD – taste, quality, freshness of dishes
2. PRICE – cost, expensiveness, value for money
3. SERVICE – staff behavior, speed, friendliness
4. AMBIENCE – atmosphere, cleanliness, environment

Sentiment Labels:
0 = Not Mentioned
1 = Negative
2 = Positive

Instructions:
1. The review may or may not be wrapped in quotation marks — ignore them, analyze only the content.
2. Determine whether each aspect is mentioned.
3. If mentioned, decide whether the sentiment is positive or negative.

Important rules:
- If the aspect is NOT discussed → 0
- If the aspect is discussed negatively → 1
- If the aspect is discussed positively → 2
- Multiple aspects may appear in one sentence.

Return ONLY a Python list with four numbers.

Order:
[FOOD, PRICE, SERVICE, AMBIENCE]
Examples:

Review: 음식은 훌륭했지만 가격이 너무 비쌌다.
Output: [2,1,0,0]

Review: 직원이 친절했고 매장 분위기도 좋았다.
Output: [0,0,2,2]

Review: 서비스는 친절하고 분위기도 아늑했지만 가격 대비 음식이 아쉬웠어요.
Output: [1,1,2,2]

Review: 가격이 너무 비싸다.
Output: [0,1,0,0]

Review: "가격이 저렴하다고 느끼긴 했지만 매장이 너무 산만하고 직원도 친절하지 않았어요."
Output: [0,2,1,1]

Review: "가격은 나쁘지 않았지만 직원 응대와 매장 환경은 불편했어요."
Output: [0,2,1,1]

Review: "장소가 아름답지 않더라도, 음식 자체가 모든 것을 말해줍니다."
Output: [2,0,0,1]

Review: {review_text}
Output:
"""