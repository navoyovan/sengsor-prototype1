#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub scale: f32,
    pub roll: f32,
    pub anchor: [f32; 2],
}

impl Transform {
    /// Computes the transformation from two eye points and a forehead anchor point.
    /// Points are in mirrored screen coordinates.
    /// Guaranteed order-invariant: sorts eyes by x coordinate so `a` is left, `b` is right.
    pub fn from_screen_points(eye_1: [f32; 2], eye_2: [f32; 2], anchor: [f32; 2]) -> Self {
        let (a, b) = if eye_1[0] <= eye_2[0] {
            (eye_1, eye_2)
        } else {
            (eye_2, eye_1)
        };

        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let scale = (dx * dx + dy * dy).sqrt();
        let roll = dy.atan2(dx);

        Self {
            scale,
            roll,
            anchor,
        }
    }

    /// Transforms a local model coordinate (where 1.0 = inter-eye distance) to screen coordinates.
    pub fn transform_point(&self, local: [f32; 2]) -> [f32; 2] {
        let cos_r = self.roll.cos();
        let sin_r = self.roll.sin();

        let sx = local[0] * self.scale;
        let sy = local[1] * self.scale;

        let rx = cos_r * sx - sin_r * sy;
        let ry = sin_r * sx + cos_r * sy;

        [self.anchor[0] + rx, self.anchor[1] + ry]
    }
}

/// Returns the local polylines for the wireframe crown.
pub fn crown_polylines() -> Vec<Vec<[f32; 2]>> {
    let outline = vec![
        [-1.2, 0.0],
        [-1.2, -0.8],
        [-0.6, -0.4],
        [0.0, -1.0],
        [0.6, -0.4],
        [1.2, -0.8],
        [1.2, 0.0],
        [-1.2, 0.0],
    ];

    let tick = vec![[0.0, 0.0], [0.0, -0.35]];

    vec![outline, tick]
}

/// Generates screen-space line segments for the transformed crown wireframe.
pub fn transform_crown(transform: &Transform) -> Vec<([f32; 2], [f32; 2])> {
    let polylines = crown_polylines();
    let mut segments = Vec::new();

    for poly in polylines {
        for window in poly.windows(2) {
            let p1 = transform.transform_point(window[0]);
            let p2 = transform.transform_point(window[1]);
            segments.push((p1, p2));
        }
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_eyes_roll_zero() {
        let eye_a = [100.0, 200.0];
        let eye_b = [200.0, 200.0];
        let anchor = [150.0, 180.0];

        let tf = Transform::from_screen_points(eye_a, eye_b, anchor);
        assert!((tf.roll).abs() < 1e-5);
        assert!((tf.scale - 100.0).abs() < 1e-5);

        // "Up" vertex (0, -1) should be directly above the anchor
        let up = tf.transform_point([0.0, -1.0]);
        assert!((up[0] - anchor[0]).abs() < 1e-4);
        assert!((up[1] - (anchor[1] - 100.0)).abs() < 1e-4);
    }

    #[test]
    fn test_swapped_eye_order_invariance() {
        let eye_a = [100.0, 200.0];
        let eye_b = [200.0, 200.0];
        let anchor = [150.0, 180.0];

        let tf1 = Transform::from_screen_points(eye_a, eye_b, anchor);
        let tf2 = Transform::from_screen_points(eye_b, eye_a, anchor);

        assert!((tf1.roll - tf2.roll).abs() < 1e-5);
        assert!((tf1.scale - tf2.scale).abs() < 1e-5);
    }

    #[test]
    fn test_tilt_clockwise() {
        // b lower than a in screen coords (larger y) -> roll > 0
        let eye_a = [100.0, 190.0];
        let eye_b = [200.0, 210.0];
        let anchor = [150.0, 180.0];

        let tf = Transform::from_screen_points(eye_a, eye_b, anchor);
        assert!(tf.roll > 0.0);

        let up = tf.transform_point([0.0, -1.0]);
        // Tilted clockwise means up vector tilts right (x > anchor.x)
        assert!(up[0] > anchor[0]);
    }

    #[test]
    fn test_scaling_doubles() {
        let eye_a = [100.0, 200.0];
        let eye_b1 = [200.0, 200.0]; // scale 100
        let eye_b2 = [300.0, 200.0]; // scale 200
        let anchor = [150.0, 180.0];

        let tf1 = Transform::from_screen_points(eye_a, eye_b1, anchor);
        let tf2 = Transform::from_screen_points(eye_a, eye_b2, anchor);

        let p1 = tf1.transform_point([1.0, 1.0]);
        let p2 = tf2.transform_point([1.0, 1.0]);

        let offset1_x = p1[0] - anchor[0];
        let offset2_x = p2[0] - anchor[0];
        assert!((offset2_x - offset1_x * 2.0).abs() < 1e-4);
    }
}
