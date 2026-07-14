#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_pointers)]
#![allow(clippy::needless_return)] // tokio macro-generated code doesn't respect this

use anyhow::Result;
use turbo_tasks::{State, Vc, unmark_top_level_task_may_leak_eventually_consistent_state};
use turbo_tasks_testing::{Registration, register, run_once};

static REGISTRATION: Registration = register!();

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_boot_constant_basic() {
    run_once(&REGISTRATION, || async {
        unmark_top_level_task_may_leak_eventually_consistent_state();
        let value = boot_constant_value(7);
        assert_eq!(*value.await?, 49);
        let derived = derived_from_boot_constant(7);
        assert_eq!(*derived.await?, 50);
        anyhow::Ok(())
    })
    .await
    .unwrap();
}

/// Invalidating a boot constant task at runtime is a hard error that aborts the process
/// (the panic escapes the per-task boundary via the cell update operation). Verify the
/// abort and its message by running the scenario in a subprocess.
#[test]
fn test_boot_constant_runtime_invalidation_aborts() {
    let exe = std::env::current_exe().unwrap();
    let output = std::process::Command::new(exe)
        .args([
            "--ignored",
            "--exact",
            "abort_on_runtime_invalidation_repro",
            "--nocapture",
        ])
        .env("RUST_BACKTRACE", "0")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !output.status.success(),
        "subprocess should die from the boot_constant violation, got: {stdout}\n{stderr}"
    );
    assert!(
        stderr.contains("is marked boot_constant, but was invalidated at runtime"),
        "expected boot_constant invalidation panic, got: {stdout}\n{stderr}"
    );
}

/// Not a real test: subprocess body for `test_boot_constant_runtime_invalidation_aborts`.
/// Must die from the boot_constant violation; exits cleanly if the violation is not
/// detected so the outer test fails.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn abort_on_runtime_invalidation_repro() {
    let result = run_once(&REGISTRATION, || async {
        unmark_top_level_task_may_leak_eventually_consistent_state();
        let input = *create_input().to_resolved().await?;
        input.await?.state.set(0);
        let value = boot_constant_reading_changing_cell(input);
        assert_eq!(*value.await?, 0);

        // Invalidate the intermediate task the boot constant depends on. When the
        // intermediate task re-executes with a different value, propagating the change to
        // the boot constant task must be a hard error (process abort).
        input.await?.state.set(1);
        // Give the invalidation propagation time to run; the process must abort before
        // this completes.
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        anyhow::Ok(())
    })
    .await;
    // Reached only if the violation was not detected.
    println!("boot_constant violation was NOT detected: {result:?}");
}

#[turbo_tasks::value]
struct ChangingInput {
    state: State<u32>,
}

#[turbo_tasks::function]
fn create_input() -> Vc<ChangingInput> {
    ChangingInput {
        state: State::new(0),
    }
    .cell()
}

#[turbo_tasks::function]
async fn read_state(input: Vc<ChangingInput>) -> Result<Vc<u32>> {
    Ok(Vc::cell(*input.await?.state.get()))
}

#[turbo_tasks::function(boot_constant)]
async fn boot_constant_reading_changing_cell(input: Vc<ChangingInput>) -> Result<Vc<u32>> {
    Ok(Vc::cell(*read_state(input).await?))
}

#[turbo_tasks::function(boot_constant)]
fn boot_constant_value(x: u32) -> Vc<u32> {
    Vc::cell(x * x)
}

#[turbo_tasks::function]
async fn derived_from_boot_constant(x: u32) -> Result<Vc<u32>> {
    Ok(Vc::cell(*boot_constant_value(x).await? + 1))
}
