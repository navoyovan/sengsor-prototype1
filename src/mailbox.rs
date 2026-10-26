use std::sync::{Arc, Condvar, Mutex};

struct Inner<T> {
    value: Option<T>,
    closed: bool,
}

/// Single-slot "latest value wins" hand-off between threads.
pub struct Mailbox<T> {
    inner: Mutex<Inner<T>>,
    cv: Condvar,
}

impl<T> Mailbox<T> {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Inner {
                value: None,
                closed: false,
            }),
            cv: Condvar::new(),
        })
    }

    /// Overwrite the slot (never blocks on the consumer).
    pub fn put(&self, v: T) {
        if let Ok(mut g) = self.inner.lock() {
            g.value = Some(v);
            self.cv.notify_one();
        }
    }

    /// Remove and return the value if present.
    pub fn try_take(&self) -> Option<T> {
        self.inner.lock().ok().and_then(|mut g| g.value.take())
    }

    /// Block until a value arrives. Returns None once closed and empty.
    pub fn take_blocking(&self) -> Option<T> {
        let mut g = self.inner.lock().ok()?;
        loop {
            if let Some(v) = g.value.take() {
                return Some(v);
            }
            if g.closed {
                return None;
            }
            g = self.cv.wait(g).ok()?;
        }
    }

    /// Empty the slot (e.g. "no face right now").
    pub fn clear(&self) {
        if let Ok(mut g) = self.inner.lock() {
            g.value = None;
        }
    }

    /// Wake all waiters; used for shutdown.
    pub fn close(&self) {
        if let Ok(mut g) = self.inner.lock() {
            g.closed = true;
            self.cv.notify_all();
        }
    }
}

impl<T: Clone> Mailbox<T> {
    /// Non-destructive read.
    pub fn get(&self) -> Option<T> {
        self.inner.lock().ok().and_then(|g| g.value.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_mailbox_overwrite_keeps_newest() {
        let mb = Mailbox::<i32>::new();
        mb.put(1);
        mb.put(2);
        mb.put(3);
        assert_eq!(mb.try_take(), Some(3));
        assert_eq!(mb.try_take(), None);
    }

    #[test]
    fn test_mailbox_take_blocking_wakes_on_put() {
        let mb = Mailbox::<String>::new();
        let mb_clone = Arc::clone(&mb);
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            mb_clone.put("hello".to_string());
        });
        assert_eq!(mb.take_blocking(), Some("hello".to_string()));
        handle.join().unwrap();
    }

    #[test]
    fn test_mailbox_close_releases_blocked_waiter() {
        let mb = Mailbox::<i32>::new();
        let mb_clone = Arc::clone(&mb);
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            mb_clone.close();
        });
        assert_eq!(mb.take_blocking(), None);
        handle.join().unwrap();
    }

    #[test]
    fn test_mailbox_clear_and_get() {
        let mb = Mailbox::<i32>::new();
        mb.put(42);
        assert_eq!(mb.get(), Some(42));
        assert_eq!(mb.get(), Some(42));
        mb.clear();
        assert_eq!(mb.get(), None);
        assert_eq!(mb.try_take(), None);
    }
}
