# -*- coding: utf-8 -*-
"""h01: JSON parser and serializer without the json module."""
import re

from common import Raises, check, forbids, load, report

forbids("rules", "myjson.py", modules=("json",))
m = load("myjson")
JsonError = getattr(m, "JsonError", None)
check("rules", "JsonError is a ValueError",
      lambda: isinstance(JsonError, type) and issubclass(JsonError, ValueError))

for text, value in [("null", None), ("true", True), ("false", False), ("0", 0), ("42", 42),
                    ("-7", -7), ("3.14", 3.14), ("1e3", 1000.0), ("-2.5E-3", -0.0025),
                    ("0.0", 0.0)]:
    check("scalars", "loads(%r)" % text, lambda t=text: m.loads(t), value)
check("scalars", "an integer stays an int", lambda: type(m.loads("42")) is int)

for text, value in [('""', ""), ('"a b"', "a b"), (r'"a\"b"', 'a"b'), (r'"\\"', "\\"),
                    (r'"\/"', "/"), (r'"\n\t\r\b\f"', "\n\t\r\b\f"), (r'"\u0041"', "A"),
                    (r'"\u010d"', u"č"), (u'"žluťoučký kůň"', u"žluťoučký kůň")]:
    check("strings", "loads(%r)" % text, lambda t=text: m.loads(t), value)

for text, value in [("[]", []), ("[1]", [1]), ("[1,2,3]", [1, 2, 3]),
                    ("[[1,[2,[3]]]]", [[1, [2, [3]]]]), ('[1,"a",null,true,[]]', [1, "a", None, True, []]),
                    ("{}", {}), ('{"a":1}', {"a": 1}), ('{"a":1,"b":[true,null]}', {"a": 1, "b": [True, None]}),
                    ('{"a":{"b":{"c":[]}}}', {"a": {"b": {"c": []}}}), ('[{"x":1},{"y":2}]', [{"x": 1}, {"y": 2}]),
                    (' \n\t{ "a" :\n [ 1 , 2 ] ,"b":\tnull }\r\n ', {"a": [1, 2], "b": None})]:
    check("containers", "loads(%r)" % text, lambda t=text: m.loads(t), value)

for value, text in [({"a": 1, "b": [True, None, "x"]}, '{"a":1,"b":[true,null,"x"]}'),
                    ([], "[]"), ({}, "{}"), (1.5, "1.5"), (False, "false"), (None, "null"),
                    ({"z": 1, "a": 2}, '{"z":1,"a":2}'), ('a"b\\c\n', r'"a\"b\\c\n"')]:
    check("dumps", "dumps(%r)" % (value,), lambda v=value: m.dumps(v), text)
check("dumps", "a control character is escaped", lambda: "\x01" not in m.dumps("\x01"))

for value in [None, True, 0, -12, 2.5, "", u"tab\tquote\"back\\slash\u0001č",
              [1, [2, [3, []]]], {"a": {"b": [None, False]}}, [{"k": "v", "n": -1.25}]]:
    check("round trip", "loads(dumps(%r))" % (value,), lambda v=value: m.loads(m.dumps(v)), value)

for text in ["{", "[1,", '{"a"}', "tru", '"unterminated', "01", '{"a":}', "[1 2]", "", "[1,]"]:
    check("errors", "loads(%r) raises JsonError" % text, lambda t=text: m.loads(t),
          Raises(JsonError or Exception))


def index_in_message():
    try:
        m.loads('[1, 2, x]')
    except Exception as e:                                # noqa: BLE001
        return bool(re.search(r"\b7\b", str(e)))
    return False


check("errors", "the message names the offending index", index_in_message)
report()
