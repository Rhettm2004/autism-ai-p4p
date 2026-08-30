import 'package:flutter/material.dart';

import '../app.dart';

class AppHeader extends StatelessWidget {
  const AppHeader({
    super.key,
    required this.stageLabel,
    required this.progress,
    required this.onMenuPressed,
    required this.onInfoPressed,
  });

  final String stageLabel;
  final double progress;
  final VoidCallback onMenuPressed;
  final VoidCallback onInfoPressed;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
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
                  tooltip: 'About this screening',
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
