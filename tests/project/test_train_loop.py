from src.training.train_loop import dummy_train_loop

def test_train_loop_epochs():
    logs = dummy_train_loop(epochs=3)
    assert len(logs) == 3
    assert all(isinstance(v, float) for v in logs)
