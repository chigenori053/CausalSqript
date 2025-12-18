from coherent.edu.cli import main as edu_main


def test_edu_cli_runs_sample(capsys):
    capsys.readouterr()
    result = edu_main(["--file", "coherent/edu/examples/pythagorean.coh"])
    captured = capsys.readouterr()
    assert result == 0
    assert "Problem" in captured.out


def test_edu_cli_runs_scenario(capsys):
    capsys.readouterr()
    result = edu_main(["--scenario", "arithmetic"])
    captured = capsys.readouterr()
    assert result == 0
    assert "Problem" in captured.out
