# Private Dataset Schema

one json one line in file

and the pretty print version here:
```json
{
    "id": "self assignd ID",
    "url": "url of the news",
    "title": "title of the news",
    "text": "content of the news",
    "word": ["word", "segmentation"],
    "entity": [[start_index, end_index, start_tag, inner_tag], [...]],
    "tag": [1,2,7],
    "category": 4,
    "sentiment": [[entity_name, score], [...]]
}
```

a simple demo
```json
{
    "id": "926",
    "url": "http://sh.sina.com.cn/news/m/2020-08-14/detail-iivhuipn8559408.shtml",
    "title": "上海今日酷热依旧最高温仍可达38℃ 未来十天高温持续",
    "text": "昨天（13）上海气温再冲新高！在副热带高压控制下，申城最高气温定格在38.6℃，成为目前为止今年最高气温，数小时稳居全国“高温榜”首位。而上一个38℃以上的高温日还是在3年前。",
    "word": ["昨天", "（", "13", "）", "上海", "气温", "再", "冲", "新高", "！", "在", "副", "热带", "高压", "控制", "下", "，", "申城", "最高", "气温", "定格", "在", "38.6", "℃", "，", "成为", "目前", "为止", "今年", "最高", "气温", "，", "数", "小时", "稳", "居", "全国", "“", "高温", "榜", "”", "首", "位", "。", "而", "上", "一个", "38", "℃", "以上", "的", "高温", "日", "还是", "在", "3", "年", "前", "。"],
    "entity": [[0, 2, "B-TIME", "I-TIME"], [6, 8, "B-LOC", "I-LOC"], [...]],
    "tag": [2, 6],
    "category": 5,
    "sentiment": [["上海", 5]]
}
```
