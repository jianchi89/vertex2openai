"""ContentDecoder 单元验证：用上游 JS 测试里的原版向量核对 Python 移植版。

只抽取 anti_truncation.py 里的 ContentDecoder 类（无外部依赖）单独执行，
因此不需要安装 fastapi / google-genai。
"""
import ast
import sys
import re
from pathlib import Path

SRC = Path(r"D:\DeepSeekH客户端\游戏安装\_v2o\vertex2openai-main\app\anti_truncation.py")
tree = ast.parse(SRC.read_text(encoding="utf-8"))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ContentDecoder")
ns = {"re": re, "List": list, "Dict": dict, "Any": object, "Optional": object}
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SRC), "exec"), ns)
ContentDecoder = ns["ContentDecoder"]

fails = []


def check(label, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + f"  {label}" + (f"   {extra}" if extra and not cond else ""))
    if not cond:
        fails.append(label)


print("1) 完整参数串（含被忽略的嵌套值、surrogate 转义、各类转义）")
args = ('{"ignored":{"list":[true,false,null,-1.25e+2]},"content":'
        '"正文\\ud83d\\ude00\\n\\"引号\\"\\\\路径\\t\\/"}')
expected = '正文😀\n"引号"\\路径\t/'
d = ContentDecoder()
out = d.feed(args)
check("正文还原正确", out == expected, f"got={out!r} want={expected!r}")
check("valid=True", d.valid is True)

print("2) 逐字符喂入（模拟上游分片边界落在任意位置）")
d = ContentDecoder()
acc = ""
for ch in args:
    acc += d.feed(ch)
check("逐字符累计一致", acc == expected, f"got={acc!r}")
check("valid=True", d.valid is True)

print("3) partialArgs 分片重组（本模块实际走的路径）")
import json as _json
fragments = ["正文", "😀", "\n", '"引', '号"', "\\", "路径", "\t", "/"]
d = ContentDecoder()
acc = d.feed('{"content":"')
for i, frag in enumerate(fragments):
    acc += d.feed(_json.dumps(frag, ensure_ascii=False)[1:-1])
    if i < len(fragments) - 1:
        pass
acc += d.feed('"')
acc += d.feed("}")
check("分片重组正文一致", acc == expected, f"got={acc!r}")
check("valid=True", d.valid is True)

print("4) 半截参数（被截断的流）：正文要能交出去，但 valid=False")
for partial in ['{"content":"open', '{"content":"partial']:
    d = ContentDecoder()
    text = d.feed(partial)
    check(f"{partial!r} 已交付文本", text == partial.split('"')[-1], f"got={text!r}")
    check(f"{partial!r} valid=False", d.valid is False)

print("5) 必须判为非法/不完整（不能当成成功）")
for bad, why in [
    ('{"content":123}', "content 不是字符串"),
    ('{"content":"a","content":"b"}', "重复 content 键"),
    ('{"content":"bad\\x"}', "非法转义"),
    ('{"ignored":' + "[" * 70 + "0" + "]" * 70 + ',"content":"ok"}', "嵌套超过 64 层"),
    ('{"content":"unterminated', "字符串未闭合"),
]:
    d = ContentDecoder()
    d.feed(bad)
    check(f"{why} → 不 valid", d.valid is False, f"invalid={d.invalid} complete={d.complete}")

print("6) 大正文（8 万字）完整还原")
value = "文" * 80000
d = ContentDecoder()
text = d.feed(_json.dumps({"content": value}, ensure_ascii=False))
check("长度一致", len(text) == 80000 and text == value, f"len={len(text)}")
check("valid=True", d.valid is True)

print("7) 无孤立代理项泄漏（保证 SSE 能被 UTF-8 编码）")
d = ContentDecoder()
text = d.feed('{"content":"\\ud83d"}')
check("孤立高代理项已替换", all(not (0xD800 <= ord(c) <= 0xDFFF) for c in text), f"got={text!r}")
try:
    text.encode("utf-8")
    ok = True
except UnicodeEncodeError:
    ok = False
check("输出可 UTF-8 编码", ok)

print("8) content 之前出现的其他字符串不得被当成正文交付")
d = ContentDecoder()
out = d.feed('{"content":"正文","note":"备注"}')
check("只交付 content", out == "正文", f"got={out!r}")
check("valid=True", d.valid is True)

print()
if fails:
    print(f"❌ {len(fails)} 项未通过：" + ", ".join(fails))
    sys.exit(1)
print("✅ 全部通过")
