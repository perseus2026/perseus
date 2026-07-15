import json
import typing as t

from cyclopts import Parameter, Token


def parse_json_mapping(_: t.Any, tokens: t.Sequence[Token]) -> dict[str, t.Any]:
    if len(tokens) != 1:
        raise ValueError("can work only with single string")
    return json.loads(tokens[0].value)


type JsonMapping = t.Annotated[dict[str, t.Any], Parameter(converter=parse_json_mapping)]
