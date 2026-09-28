def test_project_importable():
    import src

    assert src.__doc__
