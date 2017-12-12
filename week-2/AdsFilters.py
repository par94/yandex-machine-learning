get_channel = {
    'paid': 'ga:sourceMedium=~google / cpc,ga:sourceMedium=~facebook / paid',
    'paid_adwords': 'ga:sourceMedium=~google / cpc',
    'paid_facebook': 'ga:sourceMedium=~facebook / paid',
    'crm': 'ga:source==newsletter,ga:medium==email',
    'direct': 'ga:source==(direct)',
    'organic': 'ga:sourceMedium=~google / organic',
    'referral_fb': 'ga:sourceMedium=~facebook.com / referral',
    'post_fb': 'ga:sourceMedium=~facebook / post',
    'other': 'ga:sourceMedium!~google / cpc;ga:sourceMedium!~facebook / paid;ga:source!=newsletter;ga:medium!=email;ga:source!=(direct);ga:sourceMedium!~google / organic;ga:sourceMedium!~facebook.com / referral;ga:sourceMedium!~facebook / post'
}
get_event = {
    #leads
    'total_lead': 'ga:eventCategory==Lead Success',
    'phone_lead': 'ga:eventCategory==Lead Success;ga:eventAction==Phone',
    'sms_lead': 'ga:eventCategory==Lead Success;ga:eventAction==SMS',
    'email_lead': 'ga:eventCategory==Lead Success;ga:eventAction==Mail',
    #post ad
    'post_ad': 'ga:eventCategory==Post Ad',
    'post_ad_init': 'ga:eventCategory==Post Ad Initiation',
    #download page
    'download_render': 'ga:eventCategory==Download Page;ga:eventAction==gtm.js,ga:eventAction==Render',
    'app_downloads': 'ga:eventCategory==Download Page;ga:eventAction==apk-download,ga:eventAction==playstore-download',
    'direct_downloads': 'ga:eventCategory==Download Page;ga:eventAction==apk-download',
    'playstore_downloads': 'ga:eventCategory==Download Page;ga:eventAction==playstore-download',
    #VIP page
    'vip_render': 'ga:eventCategory==VIP Page;ga:eventAction==Render',
    #To add VIP activations after data-layer fix
    #Product page view
    'product_pageview': 'ga:eventCategory==Product Page View'
}
get_usertype = {
    'returning_users': 'ga:userType==Returning Visitor',
    'new_users': 'ga:userType==New Visitor'
}
get_devicetype = {
    'mobile': 'ga:deviceCategory==mobile;ga:dimension2!@gonative', #excluding app traffic
    'desktop': 'ga:deviceCategory==desktop',
    'tablet': 'ga:deviceCategory==tablet',
    'app': 'ga:dimension2=@gonative' #Includes only traffic coming from the app. User-agent used as a custom dimension
}
