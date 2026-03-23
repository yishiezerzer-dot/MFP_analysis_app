// default_microscopy_pipeline.ijm
// Args: input=<path> output=<dir>
// This is a minimal example pipeline you can customize.

requires("1.53");

function getArgValue(key) {
    // ImageJ macro args arrive as a single string: "k=v k2=v2"
    a = getArgument();
    if (a=="" || a==null) return "";
    parts = split(a, " ");
    for (i=0; i<parts.length; i++) {
        kv = split(parts[i], "=");
        if (kv.length>=2 && kv[0]==key) {
            // Re-join in case value had '=' (rare)
            v = kv[1];
            for (j=2; j<kv.length; j++) v = v + "=" + kv[j];
            return v;
        }
    }
    return "";
}

inputPath = getArgValue("input");
outputDir = getArgValue("output");

if (inputPath=="" || outputDir=="") {
    print("ERROR: Missing args. Expected: input=<path> output=<dir>");
    exit();
}

// Ensure output folder exists
File.makeDirectory(outputDir);

// Open and do a tiny example workflow
open(inputPath);

// Basic contrast
run("Enhance Contrast", "saturated=0.35");

// Example: threshold + particle analysis (works for some images; customize as needed)
// Duplicate so we don't modify original display
run("Duplicate...", "title=tmp_for_analysis");
setAutoThreshold("Default");
run("Convert to Mask");
run("Analyze Particles...", "size=50-Infinity summarize add");

// Save results table (if any)
if (isOpen("Results")) {
    saveAs("Results", outputDir + File.separator + "results.csv");
}

// Save an overlay/snapshot of the original image
selectWindow(getTitle());
saveAs("PNG", outputDir + File.separator + "overlay.png");

// Cleanup
run("Close All");
