from collections import deque


class SingleNode():
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()
        self.children = []

    def is_leaf(self):
        if self.is_str:
            return True
        return len(self.node.children()) == 0

    def get_token(self, lower=True):
        if self.is_str:
            return self.node
        name = self.node.__class__.__name__
        token = name
        is_name = False
        if self.is_leaf():
            attr_names = self.node.attr_names
            if attr_names:
                if 'names' in attr_names:
                    token = self.node.names[0]
                elif 'name' in attr_names:
                    token = self.node.name
                    is_name = True
                else:
                    token = self.node.value
            else:
                token = name
        else:
            if name == 'TypeDecl':
                token = self.node.declname
            if self.node.attr_names:
                attr_names = self.node.attr_names
                if 'op' in attr_names:
                    if self.node.op[0] == 'p':
                        token = self.node.op[1:]
                    else:
                        token = self.node.op
        if token is None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token



class TreeTraverse():
    def __init__(self):
        pass

    def preorder(self, node):
        def _preorder(node):
            current = SingleNode(node)
            sequence.append(current.get_token())
            for _, child in node.children():
                _preorder(child)
            if current.get_token().lower() == 'compound':
                sequence.append('End')
        
        sequence = []
        _preorder(node)
        return sequence[:]


    def postorder(self, node):
        def _postorder(node):
            current = SingleNode(node)
            sequence.append(current.get_token())
            for _, child in node.children()[::-1]:
                _postorder(child)
            if current.get_token().lower() == 'compound':
                sequence.append('End')

        sequence = []
        _postorder(node)
        return sequence[::-1]
    
    
    def levelorder(self, node):
        sequence = []
        que = deque([node])
        while que:
            node = que.popleft()
            current = SingleNode(node)
            sequence.append(current.get_token())
            for _, child in node.children():
                que.append(child)
            if current.get_token().lower() == 'compound':
                sequence.append('End')
        return sequence


if __name__ == '__main__':
    TTraverse = TreeTraverse()

    import pandas as pd
    df = pd.read_pickle("./data/test_.pkl")
    ast = df['code'][0]
    
    # src = r'''
    #     int main(void)
    #     {
    #         int p = 1;
    #     }
    #     '''
    # from pycparser import c_parser
    # parser = c_parser.CParser()
    # ast = parser.parse(src)

    ast.show()

    pre_seq = TTraverse.preorder(ast)
    print(len(pre_seq))
    print(pre_seq)
    print()

    pos_seq = TTraverse.postorder(ast)
    print(len(pos_seq))
    print(pos_seq)
    print()

    level_seq = TTraverse.levelorder(ast)
    print(len(level_seq))
    print(level_seq)

    a = set(pos_seq)
    b = set(pre_seq)
    c = set(pos_seq)
    print(a==b, b==c, c==a)