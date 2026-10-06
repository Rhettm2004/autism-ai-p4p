import 'package:flutter/material.dart';

import '../app.dart';

class AppHeader extends StatelessWidget {
  const AppHeader({
    super.key,
    required this.stageLabel,
    required this.progress,
    required this.onMenuPressed,
    required this.onInfoPressed,
    required this.onNewScreening,
  });

  final String stageLabel;
  final double progress;
  final VoidCallback onMenuPressed;
  final VoidCallback onInfoPressed;
  final VoidCallback onNewScreening;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: double.infinity,
            height: 58,
            child: Row(
              children: [
                IconButton(
                  onPressed: onMenuPressed,
                  tooltip: 'Open menu',
                  icon: const Icon(Icons.menu_rounded),
                ),
                const Icon(
                  Icons.all_inclusive_rounded,
                  color: AppColors.blue,
                  size: 30,
                  semanticLabel: 'Autism AI logo',
                ),
                const SizedBox(width: 8),
                const Text.rich(
                  TextSpan(
                    children: [
                      TextSpan(
                        text: 'Autism ',
                        style: TextStyle(
                          color: AppColors.navy,
                          fontSize: 18,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      TextSpan(
                        text: 'AI',
                        style: TextStyle(
                          color: AppColors.blue,
                          fontSize: 18,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                    ],
                  ),
                ),
                const Spacer(),
                if (MediaQuery.sizeOf(context).width >= 700)
                  TextButton.icon(
                    key: const Key('new-screening-header'),
                    onPressed: onNewScreening,
                    icon: const Icon(Icons.add_circle_outline),
                    label: const Text('Start new screening'),
                  )
                else
                  IconButton(
                    key: const Key('new-screening-header'),
                    onPressed: onNewScreening,
                    tooltip: 'Start new screening',
                    icon: const Icon(Icons.add_circle_outline),
                  ),
                const SizedBox(width: 12),
                Flexible(
                  child: Text(
                    stageLabel,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.labelLarge?.copyWith(
                      color: AppColors.navy,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
                IconButton(
                  onPressed: onInfoPressed,
                  tooltip: 'Information about this stage',
                  icon: const Icon(Icons.info_outline_rounded),
                ),
              ],
            ),
          ),
          Semantics(
            label: 'Screening progress ${(progress * 100).round()} percent',
            child: LinearProgressIndicator(
              value: progress,
              minHeight: 4,
              color: AppColors.blue,
              backgroundColor: AppColors.softBlue,
            ),
          ),
        ],
      ),
    );
  }
}
