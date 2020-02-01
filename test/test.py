from tf_tree import TfTree
model = TfTree(vocab_dim=15, intermediate_supervision=True, hidden_dim=15)
_ = model.fit([[0.0]*60]*5, [(0,0,0)]*5)
