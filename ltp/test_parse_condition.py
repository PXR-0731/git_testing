from parse_condition import parse_sentence, parse_condition

class Test_ParseCondition:
    def setup_method(self):
        pass

    def test_parse(self):
        question = '壁厚小于5毫米的产品'
        data = parse_sentence(question)
        assert data == {'words': ['壁厚', '小于', '5', '毫米', '的', '产品'], 'tags': ['a', 'v', 'm', 'q', 'u', 'n'], 'arcs': [(2, 'SBV'), (0, 'HED'), (4, 'ATT'), (2, 'VOB'), (4, 'RAD'), (2, 'VOB')]}
        condition = parse_condition(question, data)
        assert condition == {'subject': ['壁厚'], 'option': ['<'], 'value': ['5'], 'quantifier': ['毫米']}

    def shutdown_method(self):
        pass