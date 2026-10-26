/// Draws a filled circle (dot) onto an RGBA8 image buffer with bounds checking.
pub fn dot(buf: &mut [u8], w: u32, h: u32, cx: f32, cy: f32, r: i32, rgba: [u8; 4]) {
    let cx_i = cx.round() as i32;
    let cy_i = cy.round() as i32;
    let r_sq = r * r;

    let min_x = (cx_i - r).max(0) as u32;
    let max_x = (cx_i + r).min(w as i32 - 1).max(0) as u32;
    let min_y = (cy_i - r).max(0) as u32;
    let max_y = (cy_i + r).min(h as i32 - 1).max(0) as u32;

    for y in min_y..=max_y {
        let dy = y as i32 - cy_i;
        let row_offset = (y * w * 4) as usize;
        for x in min_x..=max_x {
            let dx = x as i32 - cx_i;
            if dx * dx + dy * dy <= r_sq {
                let idx = row_offset + (x * 4) as usize;
                buf[idx] = rgba[0];
                buf[idx + 1] = rgba[1];
                buf[idx + 2] = rgba[2];
                buf[idx + 3] = rgba[3];
            }
        }
    }
}

/// Draws a line between `a` and `b` onto an RGBA8 image buffer using Bresenham's algorithm with thickness.
pub fn line(
    buf: &mut [u8],
    w: u32,
    h: u32,
    a: [f32; 2],
    b: [f32; 2],
    thickness: i32,
    rgba: [u8; 4],
) {
    let mut x0 = a[0].round() as i32;
    let mut y0 = a[1].round() as i32;
    let x1 = b[0].round() as i32;
    let y1 = b[1].round() as i32;

    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let half_th = thickness / 2;

    loop {
        // Stamp a square around (x0, y0)
        let min_x = (x0 - half_th).max(0) as u32;
        let max_x = (x0 + half_th).min(w as i32 - 1).max(0) as u32;
        let min_y = (y0 - half_th).max(0) as u32;
        let max_y = (y0 + half_th).min(h as i32 - 1).max(0) as u32;

        for y in min_y..=max_y {
            let row_offset = (y * w * 4) as usize;
            for x in min_x..=max_x {
                let idx = row_offset + (x * 4) as usize;
                buf[idx] = rgba[0];
                buf[idx + 1] = rgba[1];
                buf[idx + 2] = rgba[2];
                buf[idx + 3] = rgba[3];
            }
        }

        if x0 == x1 && y0 == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_within_bounds() {
        let mut buf = vec![0u8; 10 * 10 * 4];
        dot(&mut buf, 10, 10, 5.0, 5.0, 1, [255, 0, 0, 255]);

        let center_idx = (5 * 10 + 5) * 4;
        assert_eq!(buf[center_idx], 255);
        assert_eq!(buf[center_idx + 1], 0);
        assert_eq!(buf[center_idx + 3], 255);
    }

    #[test]
    fn test_line_drawing() {
        let mut buf = vec![0u8; 10 * 10 * 4];
        line(
            &mut buf,
            10,
            10,
            [1.0, 1.0],
            [8.0, 1.0],
            1,
            [0, 255, 0, 255],
        );

        // All points along y = 1 from x=1..=8 should be green
        for x in 1..=8 {
            let idx = (1 * 10 + x) * 4;
            assert_eq!(buf[idx + 1], 255);
            assert_eq!(buf[idx + 3], 255);
        }
    }
}
