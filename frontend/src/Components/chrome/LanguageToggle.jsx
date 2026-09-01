import { useTranslation } from 'react-i18next';

export default function LanguageToggle() {
  const { i18n } = useTranslation();

  const toggleLanguage = () => {
    const nextLng = i18n.language === 'en' ? 'ne' : 'en';
    i18n.changeLanguage(nextLng);
  };

  return (
    <button
      type="button"
      onClick={toggleLanguage}
      className="flex items-center justify-center h-8 px-2 rounded hover:bg-[var(--accent-12)] text-ink text-sm font-medium transition-colors"
      title="Toggle Language"
    >
      {i18n.language === 'en' ? 'नेपा' : 'EN'}
    </button>
  );
}
