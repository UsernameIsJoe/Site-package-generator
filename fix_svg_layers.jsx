#target illustrator

/**
 * Promote top-level SVG groups to Illustrator layers.
 *
 * Illustrator imports SVG into a single "Layer 1". This script moves each
 * top-level group on that layer into its own document layer, named after the group.
 *
 * Usage: File > Open your SVG, then File > Scripts > Other Script... > fix_svg_layers.jsx
 */

(function () {
    if (app.documents.length === 0) {
        alert("Open the exported SVG first.");
        return;
    }

    var doc = app.activeDocument;
    if (doc.layers.length === 0) {
        alert("No layers found in the document.");
        return;
    }

    var sourceLayer = doc.layers[0];
    var groups = sourceLayer.groupItems;

    if (groups.length === 0) {
        alert("No top-level groups found on the first layer.");
        return;
    }

    for (var i = groups.length - 1; i >= 0; i--) {
        var group = groups[i];
        var newLayer = doc.layers.add();
        newLayer.name = group.name || ("Layer " + (i + 1));

        while (group.pageItems.length > 0) {
            group.pageItems[0].move(newLayer, ElementPlacement.PLACEATBEGINNING);
        }
    }

    if (sourceLayer.pageItems.length === 0 && sourceLayer.layers.length === 0) {
        sourceLayer.remove();
    }

    app.redraw();
    alert("Converted " + groups.length + " groups into Illustrator layers.");
})();
