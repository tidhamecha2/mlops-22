# examples of unit test cases

def sum(x, y):
    return x+y

def test_sum():
    x = 5
    y = 7
    z = sum(x,y)
    expected_z = 12
    assert z == expected_z

def test_equal():
    assert 1==1



