"""Generate and save the source map and both transfer-target maps.

Run this once (or any time you want to regenerate the maps from
scratch -- generation is fully deterministic given the same
student ID, so the output is identical every time).
"""

from __future__ import annotations

import argparse

from environments import generator as gen


def main(argv=None):
    """CLI entry point: generate source + transfer_similar + transfer_different maps.

    Parameters
    ----------
    argv : list of str, optional
        Argument list (defaults to ``sys.argv[1:]`` if ``None``).

    Returns
    -------
    int
        Process exit code (always 0 on success; generation failures
        raise via ``environments.generator``'s own error handling).
    """
    parser = argparse.ArgumentParser(
        description="Generate the source map and both transfer-target maps."
    )
    parser.add_argument("--student-id", type=str, default="40",
                         help="Student ID to derive base_seed/maze_size from.")
    parser.add_argument("--maps-dir", type=str, default="environments/maps",
                         help="Output directory for the generated map JSON files.")
    parser.add_argument("--similar-change-fraction", type=float, default=0.175,
                         help="Fraction of obstacles moved for the 'similar' transfer target.")
    parser.add_argument("--different-change-fraction", type=float, default=0.4,
                         help="Fraction of obstacles moved for the 'different' transfer target.")
    parser.add_argument("--different-new-penalties", type=int, default=4,
                         help="Number of new penalty cells added for the 'different' target.")
    args = parser.parse_args(argv)

    source = gen.generate_map(args.student_id, name="source")
    gen.save_map(source, args.maps_dir)
    print(f"source: maze_size={source.maze_size}, attempt={source.generation_attempt} "
          f"-> saved to {args.maps_dir}/source.json")

    similar = gen.generate_transfer_target(
        source,
        name="transfer_similar",
        change_fraction=args.similar_change_fraction,
        move_key_or_goal=False,
        n_new_penalties=0,
    )
    gen.save_map(similar, args.maps_dir)
    print(f"transfer_similar: attempt={similar.generation_attempt} "
          f"-> saved to {args.maps_dir}/transfer_similar.json")

    different = gen.generate_transfer_target(
        source,
        name="transfer_different",
        change_fraction=args.different_change_fraction,
        move_key_or_goal=True,
        n_new_penalties=args.different_new_penalties,
    )
    gen.save_map(different, args.maps_dir)
    print(f"transfer_different: attempt={different.generation_attempt} "
          f"-> saved to {args.maps_dir}/transfer_different.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
