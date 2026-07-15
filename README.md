# Perseus

A framework for training personalization models over heterogeneous sequences of user events, for any downstream task.

## Overview

Perseus turns a user's raw event history into a single embedding that describes the user at a point in time, and trains a task-specific head on top of it.
You describe everything — the events, their features, the encoders, the model backbone, the task and its metrics — in one small YAML config, and Perseus assembles the model from those pieces.

It is built for the common problem of **personalization from behavioral data**: you have long, heterogeneous sequences of events per user (purchases, clicks, transactions, ...), each with different attributes, and you want to train a model for a concrete business task without hand-building a feature pipeline every time.

Out of the box Perseus supports four task families:

- **regression** — predict a numeric target;
- **classification** — predict a class;
- **retrieval** — select candidate items for a user;
- **ranking** — order items for a user.

### Key concepts

- **Basis** — the dataset of objects. An object is a `client_id` at a given `timestamp`, optionally with `context` features, a `target`, and `groups` for sliced metrics.
- **Features & encoders** — each event/context/artifact feature is turned into an embedding by an encoder: `id` (categorical), `ple` (numerical), `bow` (text), or your own.
- **Backbone** — aggregates the event sequence (plus context) into one embedding. History aggregators include `modern_bert` (default), `bert`, `ligr`, `danet`, `hstu`, and `mamba`.
- **Task** — the head, loss, and metrics on top of the backbone output.

Encoders, aggregators, and tasks are all extensible — you can plug in custom implementations via the library interface.

## Installation

Requires Python `>=3.12,<3.13`. The project uses [uv](https://docs.astral.sh/uv/).

```bash
# clone, then:
uv sync
```

Distributed training and inference are driven by [`accelerate`](https://huggingface.co/docs/accelerate).

## Event Hub

Events are the main source of information about a user.
In this open-source release Perseus ships with its **own built-in event store** ("Event Hub") — a local, client-partitioned Parquet store — so you don't need any external event infrastructure to train a model.

### Configuring storages

An Event Hub *source* is just a directory, pointed to by an environment variable named `internal_storage_<source>`. The default source is `event_hub`:

```dotenv
# .env
internal_storage_event_hub=/path/to/event-hub
```

You can register any number of sources (e.g. `internal_storage_my_custom_events=...`) and select which one an event is read from via the `source` field in the config.
If `source` is omitted, `event_hub` is used.

### Managing events

Events are added as a Parquet table. Each table must contain a non-empty `client_id` column (`str`) and a `timestamp` column (`datetime[ns]`); any other columns become event attributes.

```bash
# add events of a given type from a parquet file
python -m perseus event-hub add-events events.pq --name transactions-purchase

# list what's available in a source (date range + attributes)
python -m perseus event-hub list-events

# delete events (optionally within a date range)
python -m perseus event-hub delete-events --name transactions-purchase \
    --date-from 2025-01-01 --date-to 2025-06-30
```

All commands accept `--source <name>` to target a non-default storage.

### Referencing events in a config

```yaml
events:
  transactions-purchase:        # read from the default `event_hub` source
    attributes:
      mcc:
      items:
        multi: true             # one event carries a set of values
    max_tokens: 25
  my-custom-event:
    attributes:
      entity_id:
    max_events_per_sequence: 100
    max_duration_per_sequence: 30d
    source: my_custom_events    # read from a custom storage
```

## Quickstart

### 1. Describe the model in `config.yaml`

```yaml
task:
  type: regression
  metrics:
    rmse:
    mae:
  preprocessor:
    scaler: log
events:
  transactions-purchase:
    attributes:
      mcc:
      total_sum:
max_events_per_sequence: 512
features:
  mcc:
    located_in:
      event: true
    encoder:
      type: id
      preprocessor:
        min_count: 100
  total_sum:
    located_in:
      event: true
    encoder:
      type: ple
backbone:
  dim: 256
  history_aggregator:
    type: modern_bert
    params:
      num_layers: 4
      num_heads: 4
```

### 2. Train

Lay out a working directory with the config and a basis split into `train`/`test` folds:

```text
train/
  config.yaml
  basis/
    train/samples.pq
    test/samples.pq
```

```bash
python -m perseus train prepare-dataset --workdir train/
accelerate launch -m perseus train fit-model --workdir train/
```

The trained checkpoint is written to `checkpoint/`.

### 3. Infer

```text
inference/
  basis/
    samples.pq
  checkpoint/
```

```bash
python -m perseus inference distribute-samples --workdir inference/
python -m perseus inference make-backbone-embeddings --workdir inference/
python -m perseus inference make-head-predictions --workdir inference/
```

Predictions land in `predictions/`.

## FAQ

**Is Perseus an end-to-end framework?**
It is primarily a sequence model over events.
External tabular features can be supplied via `context`, which is enough to train competitive end-to-end models, but that is not the main focus.

**How do I split data for training?**
You prepare the `train`/`test` folds yourself. A fixed-time temporal split is recommended to avoid leakage and get realistic metrics.

**Can I add my own encoder / aggregator / task?**
Yes. All three are extensible through the library interface.

## License

Licensed under the [Apache License 2.0](LICENSE).
