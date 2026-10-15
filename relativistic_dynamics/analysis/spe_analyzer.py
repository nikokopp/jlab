import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime

from pathlib import Path

def read_spe(filename):
    """Read .spe file, including header and footer data."""
    path = Path(filename)
    lines = path.read_text().splitlines()

    i_time = lines.index("$MEAS_TIM:")
    live_time_s, real_time_s = map(float, lines[i_time + 1].split())

    i_data = lines.index("$DATA:")
    first_channel, last_channel = map(int, lines[i_data + 1].split())

    n_channels = last_channel - first_channel + 1
    channels = np.arange(first_channel, last_channel + 1)
    
    i_counts = i_data + 2
    counts = np.array([int(line.strip()) for line in lines[i_counts : i_counts + n_channels]], dtype=int)

    return {
        "path": path,
        "channels": channels,
        "counts": counts,
        "live_time_s": live_time_s,
        "real_time_s": real_time_s
    }

def parse_run_name(filename):
    """Extract a calibration run date or velocity sweep run settings."""
    stem = Path(filename).stem

    date_match = re.search(r'(?<!\d)(20\d{6})(?!\d)', stem)
    date = None
    if date_match:
        try:
            date = datetime.strptime(date_match.group(1), '%Y%m%d').date().isoformat()
        except ValueError:
            pass

    pattern = (
        r"B(?P<B>\d+)G_"
        r"Vsel(?P<v_integer>\d+)p(?P<v_decimal>\d+)kV_"
        r"r(?P<run>\d+)"
    )
    match = re.search(pattern, stem)

    if match:
        return {
            'B_G': float(match.group('B')),
            'Vsel_kV': float(f"{match.group('v_integer')}.{match.group('v_decimal')}"),
            'run': int(match.group('run')),
            'date': date,
        }

    if date and 'calibration' in stem.lower():
        return {'date': date, 'type': 'calibration'}

    return None

def roi_stats(spectrum, channel_min, channel_max):
    """
    Calculate useful statistics inside a chosen channel region (region of interest, aka roi).
    """
    channels = spectrum["channels"]
    counts = spectrum["counts"]
    live_time = spectrum["live_time_s"]

    mask = (channels >= channel_min) & (channels <= channel_max)

    roi_channels = channels[mask]
    roi_counts = counts[mask]

    total_counts = int(np.sum(roi_counts))
    count_rate = total_counts / live_time
    count_rate_error = np.sqrt(total_counts) / live_time

    if total_counts == 0:
        median_channel = np.nan
    else:
        cumulative_counts = np.cumsum(roi_counts)
        median_index = np.searchsorted(cumulative_counts, total_counts / 2)

        median_channel = roi_channels[median_index]

    return {
        "total_counts": total_counts,
        "count_rate_Hz": count_rate,
        "count_rate_error_Hz": count_rate_error,
        "median_channel": median_channel
    }

def plot_spectrum(spectrum):
    plt.figure()

    plt.step(
        spectrum["channels"],
        spectrum["counts"],
        where="mid",
    )

    plt.xlabel("Channel")
    plt.ylabel("Counts")
    plt.title(spectrum["path"].name)
    plt.tight_layout()
    plt.show()

def _load_spectrum(spectrum):
    if isinstance(spectrum, (str, Path)):
        return read_spe(spectrum)
    return spectrum


def _check_matching_channels(first, second):
    if not np.array_equal(first['channels'], second['channels']):
        raise ValueError('Spectra have different MCA channel grids.')

    if first['live_time_s'] <= 0 or second['live_time_s'] <= 0:
        raise ValueError('Spectrum live times must be positive.')


def sum_spectra(*spectra):
    """Add raw counts and acquisition times from consecutive compatible runs."""
    if len(spectra) < 2:
        raise ValueError('Provide at least two spectra to sum.')

    spectra = [_load_spectrum(s) for s in spectra]
    first = spectra[0]

    counts = np.zeros_like(first['counts'], dtype=np.int64)
    live_time = 0.0
    real_time = 0.0
    sources = []

    for spectrum in spectra:
        _check_matching_channels(first, spectrum)
        current = np.asarray(spectrum['counts'])

        if np.any(current < 0) or not np.all(current == np.floor(current)):
            raise ValueError('Summation requires nonnegative integer counts.')

        counts += current.astype(np.int64)
        live_time += float(spectrum['live_time_s'])
        real_time += float(spectrum['real_time_s'])
        sources.append(str(spectrum['path']))

    return {
        **first,
        'counts': counts,
        'live_time_s': live_time,
        'real_time_s': real_time,
        'source_files': sources,
    }

def write_combined_spe(spectrum, output_path):
    """Save summed integer counts as an SPE file using the first run's header."""
    sources = spectrum.get('source_files', [])
    if not sources:
        raise ValueError('Expected a spectrum produced by sum_spectra().')

    output_path = Path(output_path)
    if any(output_path.resolve() == Path(f).resolve() for f in sources):
        raise ValueError('Do not overwrite an original acquisition.')

    lines = Path(sources[0]).read_text().splitlines()
    i_time = lines.index('$MEAS_TIM:')
    i_data = lines.index('$DATA:')
    first_channel, last_channel = map(int, lines[i_data + 1].split())

    counts = np.asarray(spectrum['counts'])
    if len(counts) != last_channel - first_channel + 1:
        raise ValueError('Combined channel count does not match the SPE header.')
    if np.any(counts < 0) or not np.all(counts == np.floor(counts)):
        raise ValueError('SPE files require nonnegative integer counts.')

    live_time = float(spectrum['live_time_s'])
    real_time = float(spectrum['real_time_s'])
    if not live_time.is_integer() or not real_time.is_integer():
        raise ValueError('SPE measurement times must be whole seconds.')

    lines[i_time + 1] = f'{int(live_time)} {int(real_time)}'
    lines[i_data + 2:i_data + 2 + len(counts)] = [f'{int(c):8d}' for c in counts]

    if '$SPEC_REM:' in lines and '$DATE_MEA:' in lines:
        i_date = lines.index('$DATE_MEA:')
        lines[i_date:i_date] = [f'COMBINED_FROM# {Path(f).name}' for f in sources]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n')
    return output_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Combine MCA SPE acquisitions.')
    subcommands = parser.add_subparsers(dest='action', required=True)

    addition = subcommands.add_parser('sum', help='Sum consecutive source runs')
    addition.add_argument('files', nargs='+')
    addition.add_argument('--output', required=True)

    args = parser.parse_args()

    if args.action == 'sum':
        combined = sum_spectra(*args.files)
        saved = write_combined_spe(combined, args.output)
        print(f'Saved {saved}')
        print(f"Live time: {combined['live_time_s']:,.0f} s; "
              f"total counts: {combined['counts'].sum():,}")
