from demo.demo_cli import main as demo_main


def test_demo_cli_runs_scenario(capsys):
    capsys.readouterr()
    result = demo_main(["--scenario", "minimal"])
    captured = capsys.readouterr()
    assert result == 0
    assert "Problem" in captured.out
