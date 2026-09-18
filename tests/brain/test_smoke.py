"""Brain smoke tests."""

from talkover.cli import build_parser


def test_parser_has_subcommands() -> None:
    parser = build_parser()
    args = parser.parse_args(["check"])
    assert args.command == "check"
