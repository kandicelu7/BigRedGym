"""Replace URDF visual meshes while preserving all physical elements."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


def replace_visual_meshes(root: ET.Element, mesh_dir: str) -> None:
    """Map each visual mesh to <stem>.obj or material-split <stem>_N.obj files."""
    directory = Path(mesh_dir)
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            stem = Path(mesh.attrib["filename"]).stem
            paths = sorted(directory.glob(f"{stem}_[0-9]*.obj"))
            if not paths:
                paths = [directory / f"{stem}.obj"]
            for path in paths:
                replacement = deepcopy(visual)
                replacement.find("geometry/mesh").set(
                    "filename", str(path.resolve(strict=True))
                )
                link.append(replacement)
            link.remove(visual)


def replace_visuals_from_mjcf(root: ET.Element, model_path: str) -> None:
    """Use an MJCF's OBJ files and explicit RGBA materials, not its dynamics.

    The donor must use the URDF's mesh stems (optionally split by material).
    Link-local transforms, collisions, joints, and inertias remain from URDF.
    """
    path = Path(model_path)
    donor = ET.parse(path).getroot()
    directory = path.parent / donor.find("compiler").get("meshdir", "")
    meshes = {
        mesh.get("name", Path(mesh.attrib["file"]).stem): Path(mesh.attrib["file"]).stem
        for mesh in donor.findall("asset/mesh")
    }
    materials = {
        material.attrib["name"]: material.attrib["rgba"]
        for material in donor.findall("asset/material")
    }
    colors = {
        meshes[geom.attrib["mesh"]]: materials[geom.attrib["material"]]
        for geom in donor.findall(".//worldbody//geom")
        if "mesh" in geom.attrib
    }
    replace_visual_meshes(root, str(directory))
    for visual in root.findall("link/visual"):
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            continue
        stem = Path(mesh.attrib["filename"]).stem
        for material in visual.findall("material"):
            visual.remove(material)
        material = ET.SubElement(visual, "material", name=f"visual_{stem}")
        ET.SubElement(material, "color", rgba=colors[stem])
