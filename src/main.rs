pub mod mailbox;
pub mod types;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    Ok(())
}
