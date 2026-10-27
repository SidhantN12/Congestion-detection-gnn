// Canvas rendering for an arbitrary ny x nx congestion grid. Values are
// written into an offscreen canvas at exactly grid resolution (one pixel
// per gcell) and then scaled onto the visible canvas with drawImage -
// this decouples draw cost from grid size (a 300x300 grid costs the same
// per-frame work as a 90x90 one: one ImageData write, one drawImage
// call), which is what makes the bloom-in animation and the zoom/pan
// affordable at any of this project's grid sizes without special-casing
// any particular dimension.

export class GridRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.buffers = {};
    this.ny = 0;
    this.nx = 0;
    this.viewport = null;
  }

  setGridShape(ny, nx) {
    this.ny = ny;
    this.nx = nx;
    this.buffers = {};
    this.viewport = { x0: 0, y0: 0, x1: nx, y1: ny };
  }

  resetViewport() {
    this.viewport = { x0: 0, y0: 0, x1: this.nx, y1: this.ny };
  }

  ensureBuffer(name) {
    if (this.buffers[name]) {
      return this.buffers[name];
    }
    const offscreen = document.createElement("canvas");
    offscreen.width = this.nx;
    offscreen.height = this.ny;
    const offscreenCtx = offscreen.getContext("2d");
    const entry = { canvas: offscreen, ctx: offscreenCtx };
    this.buffers[name] = entry;
    return entry;
  }

  // Writes an RGBA buffer from a flat value array (length ny*nx, row-major
  // y*nx+x, matching the API's gcell ordering) through colorFn(value) -> [r,g,b]
  // into the named layer (e.g. "prediction", "truth", "difference") - named
  // so Compare mode can hold prediction and truth simultaneously and draw
  // each into its own clipped half.
  buildBuffer(name, values, colorFn) {
    const entry = this.ensureBuffer(name);
    const n = this.nx * this.ny;
    const imageData = entry.ctx.createImageData(this.nx, this.ny);
    const data = imageData.data;
    for (let i = 0; i < n; i++) {
      const rgb = colorFn(values[i]);
      const offset = i * 4;
      data[offset] = rgb[0];
      data[offset + 1] = rgb[1];
      data[offset + 2] = rgb[2];
      data[offset + 3] = 255;
    }
    entry.ctx.putImageData(imageData, 0, 0);
  }

  resizeToDisplaySize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    return { width, height };
  }

  // Die-aspect-ratio-preserving destination rect, letterboxed within the
  // canvas. dieAspect = die_w / die_h.
  computeDestRect(canvasWidth, canvasHeight, dieAspect) {
    const canvasAspect = canvasWidth / canvasHeight;
    let destWidth;
    let destHeight;
    if (canvasAspect > dieAspect) {
      destHeight = canvasHeight;
      destWidth = destHeight * dieAspect;
    } else {
      destWidth = canvasWidth;
      destHeight = destWidth / dieAspect;
    }
    const destX = (canvasWidth - destWidth) / 2;
    const destY = (canvasHeight - destHeight) / 2;
    return { x: destX, y: destY, width: destWidth, height: destHeight };
  }

  clear(canvasWidth, canvasHeight) {
    this.ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  }

  // gcell y=0 is the die's physical BOTTOM (OpenROAD's bottom-left-origin
  // convention - see notes/phase2-congestion-extraction.md, "use
  // origin='lower' when plotting"), but the offscreen buffer is written
  // with data row 0 (gcell y=0) as image row 0, which is the TOP by
  // normal canvas/image convention. Every grid<->canvas conversion below
  // flips through this single helper so the displayed die orientation
  // matches the real layout instead of being upside down.
  gridYToCanvasY(gridY, destRect) {
    const vp = this.viewport;
    const viewportHeight = vp.y1 - vp.y0;
    const fracFromTop = (vp.y1 - gridY) / viewportHeight;
    return destRect.y + fracFromTop * destRect.height;
  }

  canvasYToGridY(canvasY, destRect) {
    const vp = this.viewport;
    const viewportHeight = vp.y1 - vp.y0;
    const fracFromTop = (canvasY - destRect.y) / destRect.height;
    return vp.y1 - fracFromTop * viewportHeight;
  }

  drawInto(name, destRect, clipRect) {
    const entry = this.buffers[name];
    if (!entry) {
      return;
    }
    const ctx = this.ctx;
    ctx.save();
    if (clipRect) {
      ctx.beginPath();
      ctx.rect(clipRect.x, clipRect.y, clipRect.width, clipRect.height);
      ctx.clip();
    }
    ctx.imageSmoothingEnabled = false;
    const vp = this.viewport;
    const viewportWidth = vp.x1 - vp.x0;
    const viewportHeight = vp.y1 - vp.y0;
    // drawImage with a negative height extends the destination rectangle
    // upward but does not reverse pixels. Flip the coordinate system so
    // the smallest physical y lands at the bottom, matching the overlays.
    ctx.translate(destRect.x, destRect.y + destRect.height);
    ctx.scale(1, -1);
    ctx.drawImage(
      entry.canvas,
      vp.x0, vp.y0, viewportWidth, viewportHeight,
      0, 0, destRect.width, destRect.height,
    );
    ctx.restore();
  }

  drawGridLines(destRect) {
    const vp = this.viewport;
    const viewportWidth = vp.x1 - vp.x0;
    const viewportHeight = vp.y1 - vp.y0;
    const cellPixelWidth = destRect.width / viewportWidth;
    const cellPixelHeight = destRect.height / viewportHeight;
    if (cellPixelWidth < 2 && cellPixelHeight < 2) {
      return;
    }

    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath();

    const firstCol = Math.floor(vp.x0);
    const lastCol = Math.ceil(vp.x1);
    for (let gx = firstCol; gx <= lastCol; gx++) {
      const px = destRect.x + (gx - vp.x0) * cellPixelWidth;
      ctx.moveTo(px, destRect.y);
      ctx.lineTo(px, destRect.y + destRect.height);
    }

    const firstRow = Math.floor(vp.y0);
    const lastRow = Math.ceil(vp.y1);
    for (let gy = firstRow; gy <= lastRow; gy++) {
      const py = this.gridYToCanvasY(gy, destRect);
      ctx.moveTo(destRect.x, py);
      ctx.lineTo(destRect.x + destRect.width, py);
    }
    ctx.stroke();
    ctx.restore();
  }

  // macros: [{x_dbu, y_dbu, w_dbu, h_dbu}], in the same DBU space as
  // die_w_dbu/die_h_dbu - independent of gcell resolution, since macros
  // aren't necessarily gcell-aligned. Always [] today (no macro
  // extraction exists anywhere in this pipeline yet); exercised once
  // real data carries macro geometry.
  drawMacroOutlines(destRect, macros, dieWDbu, dieHDbu) {
    if (!macros || macros.length === 0) {
      return;
    }
    const vp = this.viewport;
    const viewportWidth = vp.x1 - vp.x0;

    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
    ctx.lineWidth = 1.5;
    for (const macro of macros) {
      const gx0 = (macro.x_dbu / dieWDbu) * this.nx;
      const gy0 = (macro.y_dbu / dieHDbu) * this.ny;
      const gx1 = ((macro.x_dbu + macro.w_dbu) / dieWDbu) * this.nx;
      const gy1 = ((macro.y_dbu + macro.h_dbu) / dieHDbu) * this.ny;
      const px0 = destRect.x + (gx0 - vp.x0) * (destRect.width / viewportWidth);
      const px1 = destRect.x + (gx1 - vp.x0) * (destRect.width / viewportWidth);
      const topY = this.gridYToCanvasY(gy1, destRect);
      const bottomY = this.gridYToCanvasY(gy0, destRect);
      ctx.strokeRect(px0, topY, px1 - px0, bottomY - topY);
    }
    ctx.restore();
  }

  // Inverse of the grid->canvas mapping: canvas pixel -> integer gcell
  // coordinate, accounting for the current viewport. Returns null if the
  // point falls outside the destination rect.
  canvasPointToGrid(px, py, destRect) {
    if (px < destRect.x || px > destRect.x + destRect.width) {
      return null;
    }
    if (py < destRect.y || py > destRect.y + destRect.height) {
      return null;
    }
    const vp = this.viewport;
    const viewportWidth = vp.x1 - vp.x0;
    const fracX = (px - destRect.x) / destRect.width;
    const gx = Math.floor(vp.x0 + fracX * viewportWidth);
    const gy = Math.floor(this.canvasYToGridY(py, destRect));
    return { gx, gy };
  }

  gridRectToCanvasRect(gx0, gy0, gx1, gy1, destRect) {
    const vp = this.viewport;
    const viewportWidth = vp.x1 - vp.x0;
    const px0 = destRect.x + (gx0 - vp.x0) * (destRect.width / viewportWidth);
    const px1 = destRect.x + (gx1 - vp.x0) * (destRect.width / viewportWidth);
    const topY = this.gridYToCanvasY(gy1, destRect);
    const bottomY = this.gridYToCanvasY(gy0, destRect);
    return { x: px0, y: topY, width: px1 - px0, height: bottomY - topY };
  }
}
