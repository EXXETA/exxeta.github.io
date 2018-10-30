---
layout: page
title: About Us
permalink: /about/
---

EXXETA is an independent and medium-sized technology
and consulting firm, with a focus on the automotive, energy
and financial services sectors.

EXXETA is your Value+ Partner. EXXETA offers you a clear
added value by bringing business and IT together. We offer
holistic and innovative solutions to our clients, ranging from
industry and IT consulting services to tailored software to
the design and implementation of forward-looking strategies
and new business models.

Swing by our offices in Berlin
to discuss your data-driven use cases with our
data science and machine learning experts!

<div class="container">
    {% for item in site.authors %}
    {% assign author = item[1] %}
    <div class="row">
        <div class="col-md-2">
            <img class="rounded float-left" src="https://www.gravatar.com/avatar/{{ author.gravatar }}?s=150&d=mm&r=x" alt="{{ author.display_name }}">
        </div>
        <div class="col-md-3">
            <strong>{{ author.display_name }}</strong><br>
            <a href="mailto:{{ author.email }}">{{ author.email }}</a><br>
            <a href="tel:{{ author.phone }}">{{ author.phone }}</a><br>
            <a target="_blank" href="{{ author.linkedin }}"><i class="fab fa-linkedin fa-2x"></i></a>
        </div>
        <div class="col-md-4">
            <span class="author-description">{{ author.description }}</span>
        </div>
    </div>
    <br>
    {% endfor %}
</div>

<address>
    <strong>EXXETA AG</strong><br>
    c/o WeWork Sony Center<br>
    Kemperplatz 1<br>
    10785 Berlin<br>
</address>

<iframe width="425" height="350" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" src="https://www.openstreetmap.org/export/embed.html?bbox=13.371380295138806%2C52.50979876436803%2C13.374416555743666%2C52.51136100220606&amp;layer=mapnik&amp;marker=52.510579890228776%2C13.372898425441235" style="border: 1px solid black"></iframe><br/><small><a href="https://www.openstreetmap.org/?mlat=52.51058&amp;mlon=13.37290#map=19/52.51058/13.37290">View Larger Map</a></small>
