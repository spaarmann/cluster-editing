use std::hash;

macro_rules! append_path_log {
    ($p:expr, $s:expr, $($arg:tt)+) => (
        #[cfg(feature = "path-log")]
        {
            $p.path_log.push_str(&format!($s, $($arg)+));
        }
    )
}

macro_rules! append_path_log_dir {
    ($l:expr, $s:expr, $($arg:tt)+) => (
        #[cfg(feature = "path-log")]
        {
            $l.push_str(&format!($s, $($arg)+));
        }
    )
}

#[allow(unused)]
macro_rules! log_indent {
    ($k:expr, $l:expr, $s:expr) => (
        log::log!(log::Level::$l, concat!("{}[k={}]", $s),
            "\t".repeat((unsafe { K_MAX - $k.max(0.0) }) as usize), $k);
    );
    ($k:expr, $l:expr, $s:expr, $($arg:tt)+) => (
        log::log!($l, concat!("{}[k={}]", $s),
            "\t".repeat((unsafe { K_MAX - $k.max(0.0) }) as usize),
            $k, $($arg)+);
    )
}

macro_rules! dbg_trace_indent {
    ($t:expr, $k:expr, $s:expr) => (
        #[cfg(feature = "detailed-logs")]
        {
            log::log!(log::Level::Trace, concat!("{}[k={}]", $s),
                "\t".repeat(($t.k_max - $k.max(0.0)) as usize), $k);
        }
    );
    ($t:expr, $k:expr, $s:expr, $($arg:tt)+) => (
        #[cfg(feature = "detailed-logs")]
        {
            log::log!(log::Level::Trace, concat!("{}[k={}]", $s),
                "\t".repeat(($t.k_max - $k.max(0.0)) as usize),
                $k, $($arg)+);
        }
    )
}

macro_rules! trace_and_path_log {
    ($p:expr, $k:expr, $s:expr, $($arg:tt)+) => (
        #[cfg(feature = "detailed-logs")]
        {
            log::log!(log::Level::Trace, concat!("{}[k={}]", $s),
                "\t".repeat(($p.k_max - $k.max(0.0)) as usize),
                $k, $($arg)+);
        }
        #[cfg(feature = "path-log")]
        {
            $p.path_log.push_str(&format!(concat!($s, "\n"), $($arg)+));
        }
    )
}

/// `continue_if_not_present(g, u)` executes a `continue` statement if vertex `u` is not present in
/// Graph `g`.
macro_rules! continue_if_not_present {
    ($g:expr, $u:expr) => {
        if !$g.is_present($u) {
            continue;
        }
    };
}

pub trait InfiniteNum: Copy {
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    fn is_finite(self) -> bool;
    fn is_infinite(self) -> bool;
}

impl InfiniteNum for f32 {
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn is_infinite(self) -> bool {
        self.is_infinite()
    }
}

// This actually leads to overflowing adds and consequently breaks stuff in its current form :(
// Currently using floats again instead, but Graph and all the other code is now generic over the
// weight, if a fixed impl for this can be done it should be easy to substitute by changing the
// `Weight` type in lib.rs.
/*impl InfiniteNum for i32 {
    const INFINITY: Self = 2 * 100000000;
    const NEG_INFINITY: Self = -2 * 100000000;
    fn is_finite(self) -> bool {
        self < 100000000 && self > -100000000
    }
    fn is_infinite(self) -> bool {
        self >= 100000000 || self <= -100000000
    }
}*/

/// This is dangerous to use in general!
/// Provides an f32 wrapper that is Hash and Eq to use as a HashMap key.
/// Uses bit-by-bit equality for hashing and comparisons; be very sure you want to use this
/// and understand the problems.
#[derive(Debug)]
pub struct FloatKey(pub f32);

impl FloatKey {
    fn key(&self) -> u32 {
        self.0.to_bits()
    }
}

impl hash::Hash for FloatKey {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state)
    }
}

impl PartialEq for FloatKey {
    fn eq(&self, other: &FloatKey) -> bool {
        self.key() == other.key()
    }
}

impl Eq for FloatKey {}
