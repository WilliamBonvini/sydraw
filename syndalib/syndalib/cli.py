"""A sample CLI."""

import click
import log


@click.command()
@click.argument("feet")
def main(text: str):
    log.init()
    click.echo(text)


if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=no-value-for-parameter
