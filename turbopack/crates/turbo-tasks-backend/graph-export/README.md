# turbo-tasks graph export

Dump the resident **turbo-tasks task graph** as JSON after a build, for offline analysis of the
graph shape: topology (child + dependency edges), task types, and cells (data slots).

This is an internal diagnostic gated behind two env vars — nothing is exported unless you ask.

## Usage

Against a real Next.js app (`next build --turbopack`):

```sh
TURBO_DEP_TRACKING=1 TURBO_MEM_GRAPH_EXPORT=graph.json \
  node packages/next/dist/bin/next build --turbopack <path-to-app>
```

Or against the standalone `turbopack-cli`:

```sh
TURBO_DEP_TRACKING=1 TURBO_MEM_GRAPH_EXPORT=graph.json \
  turbopack-cli build index.js --dir <project>
```

Rebuild the native binding after changing this feature (`pnpm --filter=@next/swc build-native`, or
`pnpm build-all` for a full bootstrap).

Two env vars, both no-ops when unset:

- `TURBO_MEM_GRAPH_EXPORT=<path>` — write the graph JSON to `<path>` after the build.
- `TURBO_DEP_TRACKING=1` — build with `dependency_tracking: true` so the `deps` / `cell_deps` edges
  are populated. A one-shot build normally runs with it off (no invalidation needed), so those
  edges are empty without it.

> The export runs **after** the build, so tasks whose cells were consumed during the build (e.g.
> `parse`, whose AST cell is taken by codegen) show `cells: 0` at rest. It captures graph
> *structure*, not the mid-build memory peak.

## JSON schema

```jsonc
{ "task_count": <n>,
  "nodes": [
    { "id":        123,              // numeric task id (edges reference these)
      "ty":        "parse",          // task type (native-fn name); null for transient/driver tasks
      "transient": false,            // session-only task vs persistent graph task
      "has_output": true,            // is the task's output value resident
      "cells":     1,                // number of resident cells (data slots) the task owns
      "children":  [124, 125],       // CHILD edges: task ids this task spawned (always present)
      "deps":      [80, 91],         // DEP edges: task ids whose output/cells this read
                                     //   (only with TURBO_DEP_TRACKING=1) — the invalidation graph
      "owned_cells": [ {"ty":"turbopack_ecmascript::...::ParseResult","index":0} ],
      "cell_deps":   [ {"task":80,"ty":"...EsmAssetReference","cell":2},   // cell-precise read
                       {"task":91,"ty":null,"cell":null} ]                 // whole-output read
    } ]
}
```

`children` (spawn/ownership) and `deps` (reads/invalidation) are **two different graphs over the same
nodes**. `cell_deps` is the cell-granular form of `deps` (a consumer depends on one *specific* cell of
a producer, not the whole task) — the task-level `deps` collapses that.

## Where the code lives

- `src/backend/storage.rs` — `Storage::export_graph_nodes()` + the `GraphNodeExport` / `CellInfo` /
  `CellDepInfo` schema.
- `src/backend/mod.rs` — `TurboTasksBackend::export_graph_json(path)` +
  `maybe_export_graph_json_from_env()`.
- `src/backend/storage_schema.rs` — the `for_each_child` / `for_each_dep` / `for_each_cell` /
  `for_each_cell_dep` / `cell_count` walkers.
- `turbopack-cli/src/build/mod.rs` — the env-var wiring for `turbopack-cli`.
- `next-napi-bindings/src/next_api/project.rs` (`project_shutdown`) +
  `next/src/build/turbopack-build/impl.ts` — the env-var wiring for `next build --turbopack`.
