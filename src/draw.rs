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
}
