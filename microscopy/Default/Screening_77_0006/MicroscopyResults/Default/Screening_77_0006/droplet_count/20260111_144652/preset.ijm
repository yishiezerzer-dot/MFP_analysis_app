
    // PresetAnalysis (generated)

    // Robust args parsing: supports quoted values with spaces.
    function getKV(args, key) {
        k = key + "=";
        i = indexOf(args, k);
        if (i < 0) return "";
        i = i + lengthOf(k);
        if (substring(args, i, i+1) == "\"") {
            j = indexOf(args, "\"", i+1);
            if (j < 0) return substring(args, i+1);
            return substring(args, i+1, j);
        }
        j = indexOf(args, " " , i);
        if (j < 0) j = lengthOf(args);
        return substring(args, i, j);
}

args = getArgument();
input = getKV(args, "input");
out = getKV(args, "output");
if (input=="" || out=="") {
    print("Missing input/output args");
    exit(1);
}

// Helper: log to file
function logLine(s) {
    File.append(s+"\n", out + "/run_log.txt");
    print(s);
}

File.makeDirectory(out);
logLine("Input: " + input);
logLine("Output: " + out);

setBatchMode(true);
open(input);
title = getTitle();

// Convert to 8-bit if needed
run("8-bit");

// Threshold
method = "Otsu";
if (toLowerCase(method) == "manual") {
    setThreshold(parseFloat("50"), parseFloat("200"));
    setOption("BlackBackground", true);
    run("Convert to Mask");
} else {
    // Auto threshold (dark objects on bright background by default)
    setAutoThreshold(method + " dark");
    setOption("BlackBackground", true);
    run("Convert to Mask");
}

// Ensure binary
run("Make Binary");

    // Measurements
    run("Set Measurements...", "area mean min perimeter shape feret's redirect=None decimal=3");

    // Analyze Particles
    edgeOpt = true;
    overlayOpt = true;
    ap = "size=10-Infinity circularity=0.00-1.00";
    if (edgeOpt) ap = ap + " exclude";
    if (overlayOpt) ap = ap + " show=Overlay"; else ap = ap + " show=Nothing";
    ap = ap + " clear";
    run("Analyze Particles...", ap);

    // Save results
    saveAs("Results", out + "/results.csv");
    if (overlayOpt) {
        // Best-effort overlay snapshot
        selectWindow(title);
        saveAs("PNG", out + "/overlay.png");
    }

    close();
    run("Close All");
    setBatchMode(false);
    logLine("Done");
