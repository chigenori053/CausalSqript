from coherent.pro.cli import main as pro_main


def test_pro_cli_runs_sample(capsys):
    capsys.readouterr()
    result = pro_main(["--file", "coherent/pro/examples/polynomial_analysis.coh"])
    captured = capsys.readouterr()
    assert result == 0
    assert "Problem" in captured.out


def test_pro_cli_runs_scenario(capsys):
    capsys.readouterr()
    result = pro_main(["--mode", "causal", "--scenario", "basic"])
    captured = capsys.readouterr()
    assert result == 0
    assert "Problem" in captured.out
